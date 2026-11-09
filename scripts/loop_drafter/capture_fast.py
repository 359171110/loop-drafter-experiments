"""Capture-only execution optimizations; FP32 target and equal-length inputs."""
from functools import partial
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from .target import TargetOutput


class AppendOnlyLayer(DynamicLayer):
    """Reserve KV storage once; expose only the initialized prefix to attention."""
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.length = 0

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            shape = (*key_states.shape[:-2], self.capacity, key_states.shape[-1])
            self.keys = key_states.new_empty(shape)
            self.values = value_states.new_empty(shape)
            self.is_initialized = True
        end = self.length + key_states.shape[-2]
        if end > self.capacity:
            raise ValueError('capture cache capacity exceeded')
        self.keys[..., self.length:end, :].copy_(key_states)
        self.values[..., self.length:end, :].copy_(value_states)
        self.length = end
        return self.keys[..., :end, :], self.values[..., :end, :]

    def get_seq_length(self):
        return self.length

    def crop(self, max_length):
        raise RuntimeError('AppendOnlyLayer is for capture, not speculative rollback')


class CaptureTarget:
    def __init__(self, target, capacity, *, buffered_cache=True):
        self.target = target
        self.model = target.model
        self.device = target.device
        self.capacity = capacity
        self.buffered_cache = buffered_cache

    @torch.inference_mode()
    def forward(self, input_ids, cache=None):
        if cache is None:
            cache = DynamicCache()
            if self.buffered_cache:
                cache.layer_class_to_replicate = partial(AppendOnlyLayer, self.capacity)
        length = cache.get_seq_length()
        positions = torch.arange(length, length + input_ids.shape[1], device=input_ids.device)
        output, states, _ = self.model.model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                          cache_position=positions, position_ids=positions.unsqueeze(0))
        if output.past_key_values is not cache or len(states) != 4:
            raise RuntimeError('capture target cache/round contract changed')
        # Generation uses only the final prompt logit; retain features at all positions.
        logits = self.model.get_output_embeddings()(states[-1][:, -1:, :])
        return TargetOutput(logits, torch.stack(states, dim=2), cache)


@torch.inference_mode()
def capture_batch_fast(target, input_ids, max_new_tokens, eos_ids, *, buffered_cache=True):
    if input_ids.ndim != 2 or min(input_ids.shape) < 1 or max_new_tokens < 1:
        raise ValueError('nonempty equal-length prompts and positive output cap required')
    fast = CaptureTarget(target, input_ids.shape[1] + max_new_tokens, buffered_cache=buffered_cache)
    output = fast.forward(input_ids)
    batch, prompt_length = input_ids.shape
    if tuple(output.round_hidden.shape[:3]) != (batch,prompt_length,4):
        raise ValueError('target must expose all four round states')
    features = torch.empty((batch,prompt_length + max_new_tokens - 1,4,target.hidden_size),
                           device=input_ids.device,dtype=torch.bfloat16)
    features[:, :prompt_length].copy_(output.round_hidden)
    generated = [[] for _ in range(batch)]
    active = [True]*batch
    eos_ids = set(eos_ids)
    for step in range(max_new_tokens):
        predicted = output.logits[:, -1].argmax(-1).tolist()
        for row in range(batch):
            if active[row]:
                generated[row].append(predicted[row])
                if predicted[row] in eos_ids:
                    active[row] = False
        if not any(active) or step + 1 == max_new_tokens:
            break
        ids = [predicted[row] if active[row] else generated[row][-1] for row in range(batch)]
        output = fast.forward(torch.tensor(ids,device=input_ids.device,dtype=torch.long)[:,None],output.cache)
        features[:,prompt_length+step:prompt_length+step+1].copy_(output.round_hidden)
    result=[]
    prompt_cpu = input_ids.cpu()
    # Transfer each finished trajectory once, instead of one synchronization per token.
    for row in range(batch):
        tokens=torch.cat((prompt_cpu[row],torch.tensor(generated[row],dtype=torch.long)))
        hidden=features[row,:len(tokens)-1].to('cpu').contiguous().clone()
        result.append((tokens,hidden))
    return result


class GraphLayer(AppendOnlyLayer):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.decode = False
        self.attention_length = capacity

    def update(self, key_states, value_states, *args, **kwargs):
        if self.decode:
            position = args[0]['cache_position'] if args else kwargs['cache_kwargs']['cache_position']
            self.keys.index_copy_(2, position, key_states)
            self.values.index_copy_(2, position, value_states)
            return self.keys[..., :self.attention_length, :], self.values[..., :self.attention_length, :]
        if not self.is_initialized:
            shape = (*key_states.shape[:-2], self.capacity, key_states.shape[-1])
            self.keys = key_states.new_zeros(shape)
            self.values = value_states.new_zeros(shape)
            self.is_initialized = True
        return super().update(key_states, value_states, *args, **kwargs)


@torch.inference_mode()
def capture_batch_graph(target, input_ids, max_new_tokens, eos_ids, *, chunk_size=128):
    """Graph replay for single-token decoding; all four target loops remain FP32."""
    if not input_ids.is_cuda:
        raise ValueError('CUDA graph capture requires CUDA inputs')
    batch, length = input_ids.shape
    capacity = length + max_new_tokens
    cache = DynamicCache()
    cache.layer_class_to_replicate = partial(GraphLayer, capacity)
    prefill = CaptureTarget(target, capacity).forward(input_ids,cache)
    features = torch.empty((batch,capacity-1,4,target.hidden_size),device=input_ids.device,dtype=torch.bfloat16)
    features[:,:length].copy_(prefill.round_hidden)
    for layer in cache.layers:
        layer.decode = True
    static_ids = torch.zeros((batch,1),device=input_ids.device,dtype=torch.long)
    position = torch.tensor([length],device=input_ids.device,dtype=torch.long)
    keys = torch.arange(capacity,device=input_ids.device)
    graph = graph_logits = graph_hidden = None
    graph_end = 0

    def build_graph(end):
        mask = torch.zeros((1,1,1,end),device=input_ids.device,dtype=torch.float32)
        for layer in cache.layers:
            layer.attention_length = end
        def decode():
            mask.copy_(torch.where(keys[:end] <= position, 0., torch.finfo(torch.float32).min).view(1,1,1,-1))
            _, states, _ = target.model.model(input_ids=static_ids, attention_mask={'full_attention':mask},
                              past_key_values=cache,use_cache=True,cache_position=position,position_ids=position.unsqueeze(0))
            return target.model.get_output_embeddings()(states[-1]),torch.stack(states,dim=2)
        stream=torch.cuda.Stream(device=input_ids.device)
        stream.wait_stream(torch.cuda.current_stream(input_ids.device))
        with torch.cuda.stream(stream):
            decode()
        torch.cuda.current_stream(input_ids.device).wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits,hidden=decode()
        return graph,logits,hidden

    logits=prefill.logits
    generated=[[] for _ in range(batch)];active=[True]*batch;eos=set(eos_ids)
    for step in range(max_new_tokens):
        predicted=logits[:,-1].argmax(-1).tolist()
        for row in range(batch):
            if active[row]:
                generated[row].append(predicted[row])
                if predicted[row] in eos: active[row]=False
        if not any(active) or step+1==max_new_tokens: break
        next_ids=[predicted[row] if active[row] else generated[row][-1] for row in range(batch)]
        static_ids.copy_(torch.tensor(next_ids,device=input_ids.device)[:,None])
        position.fill_(length+step)
        if length+step >= graph_end:
            graph_end = min(capacity, length+step+chunk_size) if chunk_size else capacity
            graph,graph_logits,graph_hidden=build_graph(graph_end)
        graph.replay()
        features[:,length+step:length+step+1].copy_(graph_hidden)
        logits=graph_logits
    prompts=input_ids.cpu()
    return [(torch.cat((prompts[row],torch.tensor(generated[row],dtype=torch.long))),
             features[row,:length+len(generated[row])-1].cpu().contiguous().clone()) for row in range(batch)]
