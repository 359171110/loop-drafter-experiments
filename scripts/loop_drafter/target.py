"""Frozen four-round Ouro target and cache checks for speculative decoding.

``round_hidden[b, t, r]`` is the checkpoint's ``hidden_states_list[r]``:
the state AFTER the shared end-of-round RMSNorm, not a decoder-layer hook.
Inputs are unpadded, equally sized token sequences. Each of the four rounds
has its own KV slots, even though the decoder weights are shared.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.dynamic_module_utils import get_class_from_dynamic_module


DEFAULT_MODEL = "/models/Ouro-1.4B"


def ouro_model_class(model_dir: str | Path) -> type[nn.Module]:
    """Load local checkpoint code, adapting only that module to HF API changes."""
    cls = get_class_from_dynamic_module(
        "modeling_ouro.OuroForCausalLM", str(model_dir), local_files_only=True,
    )
    module = sys.modules[cls.__module__]
    # HF 5 removed the default entry. Do not change HF's process-wide registry.
    if "default" not in module.ROPE_INIT_FUNCTIONS:
        def default_rope(config: Any, device: Any = None, **kwargs: Any) -> tuple[torch.Tensor, float]:
            del kwargs
            dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            exponent = torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
            return 1.0 / float(config.rope_theta) ** exponent, 1.0

        module.ROPE_INIT_FUNCTIONS = {**module.ROPE_INIT_FUNCTIONS, "default": default_rope}

    rotary = module.OuroRotaryEmbedding
    if not hasattr(rotary, "compute_default_rope_parameters"):
        def compute_default_rope_parameters(self: nn.Module, config: Any = None) -> tuple[torch.Tensor, float]:
            return module.ROPE_INIT_FUNCTIONS["default"](self.config if config is None else config)

        rotary.compute_default_rope_parameters = compute_default_rope_parameters

    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        original = getattr(module, name)
        if getattr(original, "_loop_drafter_compatible", False):
            continue
        parameters = inspect.signature(original).parameters

        def compatible_mask(*args: Any, _original: Any = original,
                            _parameters: Any = parameters, **kwargs: Any) -> Any:
            if "inputs_embeds" in _parameters and "input_embeds" in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            if "cache_position" not in _parameters:
                kwargs.pop("cache_position", None)
            return _original(*args, **kwargs)

        compatible_mask._loop_drafter_compatible = True
        setattr(module, name, compatible_mask)
    return cls


@dataclass
class TargetOutput:
    """One target forward; logits predict the token following each input token."""

    logits: torch.Tensor
    round_hidden: torch.Tensor
    cache: DynamicCache

    @property
    def past_key_values(self) -> DynamicCache:
        """Expose the conventional Transformers cache name."""
        return self.cache


class OuroTarget:
    """Run the actual four-loop backbone and project only its final-round state."""

    rounds = 4
    num_loops = 4

    def __init__(self, model: nn.Module, tokenizer: Any = None) -> None:
        if getattr(model.config, "model_type", None) != "ouro":
            raise ValueError("OuroTarget requires an Ouro checkpoint")
        if int(model.config.total_ut_steps) != self.rounds or int(model.model.total_ut_steps) != self.rounds:
            raise ValueError("The target must actually execute exactly four rounds")
        if any(kind != "full_attention" for kind in model.config.layer_types):
            raise ValueError("This target adapter currently requires full-attention Ouro layers")
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        self.hidden_size = int(model.config.hidden_size)
        self.vocab_size = int(model.config.vocab_size)
        self.logical_layers = self.rounds * int(model.config.num_hidden_layers)

    @classmethod
    def from_pretrained(
        cls, model_dir: str | Path = DEFAULT_MODEL, *, device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16, attention_backend: str = "sdpa",
    ) -> "OuroTarget":
        """Load only local files and freeze the original checkpoint parameters."""
        if attention_backend not in ("sdpa", "eager"):
            raise ValueError("attention_backend must be sdpa or eager")
        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
        if int(config.total_ut_steps) != 4:
            raise ValueError("The supplied checkpoint is not configured for four actual rounds")
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = config.eos_token_id
        model_cls = ouro_model_class(model_dir)
        model = model_cls.from_pretrained(
            str(model_dir), config=config, dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation=attention_backend, local_files_only=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
        return cls(model, tokenizer)

    @property
    def device(self) -> torch.device:
        """Device holding the input embedding table."""
        return self.model.get_input_embeddings().weight.device

    @staticmethod
    def new_cache() -> DynamicCache:
        """A growable cache, without a physical-layer-only config."""
        return DynamicCache()

    fresh_cache = new_cache

    def cache_lengths(self, cache: DynamicCache) -> list[int]:
        """Lengths of all logical attention slots, including empty slots."""
        return [int(cache.get_seq_length(index)) for index in range(self.logical_layers)]

    def _check_cache(self, cache: DynamicCache, expected: int) -> None:
        if not isinstance(cache, DynamicCache):
            raise TypeError("Use a growable DynamicCache for the four-round target")
        lengths = self.cache_lengths(cache)
        if any(length != expected for length in lengths):
            raise ValueError(f"Logical cache lengths do not all equal {expected}: {lengths}")

    def clone_cache(self, cache: DynamicCache) -> DynamicCache:
        """Copy KV tensors so speculative mutation cannot change the source cache."""
        self._check_cache(cache, int(cache.get_seq_length()))
        return copy.deepcopy(cache)

    def crop_cache(self, cache: DynamicCache, prefix_length: int) -> DynamicCache:
        """Keep only committed tokens in every round's KV slots, in place."""
        current = int(cache.get_seq_length())
        if not 0 <= prefix_length <= current:
            raise ValueError(f"prefix_length must be between 0 and {current}")
        self._check_cache(cache, current)
        cache.crop(prefix_length)
        self._check_cache(cache, prefix_length)
        return cache

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, cache: DynamicCache | None = None) -> TargetOutput:
        """Append nonempty tokens and return final logits plus normalized round states."""
        if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have nonempty shape [batch, tokens], without padding")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must be torch.long")
        if self.model.training or any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Target must remain in eval mode with every parameter frozen")
        if self.model.model.total_ut_steps != 4 or self.model.config.total_ut_steps != 4:
            raise RuntimeError("Target round count was changed after construction")
        cache = self.new_cache() if cache is None else cache
        previous_length = int(cache.get_seq_length())
        self._check_cache(cache, previous_length)
        input_ids = input_ids.to(self.device)
        positions = torch.arange(previous_length, previous_length + input_ids.shape[1], device=self.device)
        output, round_states, _ = self.model.model(
            input_ids=input_ids, past_key_values=cache, use_cache=True,
            cache_position=positions, position_ids=positions.unsqueeze(0),
        )
        if output.past_key_values is not cache:
            raise RuntimeError("Checkpoint replaced the explicitly supplied DynamicCache")
        if len(round_states) != self.rounds:
            raise RuntimeError("The backbone did not return all four round states")
        round_hidden = torch.stack(round_states, dim=2)
        expected_shape = (*input_ids.shape, self.rounds, self.hidden_size)
        if tuple(round_hidden.shape) != expected_shape:
            raise RuntimeError(f"Unexpected round state shape {tuple(round_hidden.shape)}")
        logits = self.model.get_output_embeddings()(round_states[-1])
        self._check_cache(cache, previous_length + input_ids.shape[1])
        return TargetOutput(logits, round_hidden, cache)

    __call__ = forward


@torch.no_grad()
def greedy_generate(target: OuroTarget, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Generate a fixed-length greedy continuation; EOS is intentionally not a stop."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    output = target(input_ids)
    tokens = []
    for index in range(max_new_tokens):
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        tokens.append(token)
        if index + 1 < max_new_tokens:
            output = target(token, output.cache)
    return torch.cat(tokens, dim=1) if tokens else input_ids.new_empty((input_ids.shape[0], 0))


def _comparison(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict[str, Any]:
    difference = (actual.float() - expected.float()).abs()
    actual_argmax, expected_argmax = actual.argmax(-1), expected.argmax(-1)
    mismatch_positions = actual_argmax.ne(expected_argmax).nonzero(as_tuple=False)
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "close": bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)),
        "argmax_equal": len(mismatch_positions) == 0,
        "argmax_mismatch_count": len(mismatch_positions),
        "argmax_mismatches": [
            {"position": index.tolist(), "actual": int(actual_argmax[tuple(index)].item()),
             "expected": int(expected_argmax[tuple(index)].item())}
            for index in mismatch_positions[:20]
        ],
    }


@torch.no_grad()
def validate_target(
    target: OuroTarget, prompts: Sequence[torch.Tensor] | torch.Tensor, *, new_tokens: int = 8,
    block_size: int = 4, atol: float = 0.05, rtol: float = 0.01,
) -> dict[str, Any]:
    """Compare serial/block execution and reject/crop/recover on small prompts.

    This is a correctness diagnostic, not a speed benchmark. A 2-D prompt
    tensor is split into unpadded single-request prompts; a list permits
    different lengths. Results include exact greedy agreement independently
    of numerical tolerances. No weights are loaded by this function.
    """
    if new_tokens < 2 or block_size < 2:
        raise ValueError("Use at least two new tokens and a block size of at least two")
    if isinstance(prompts, torch.Tensor):
        prompts = [row.unsqueeze(0) for row in prompts]
    if not prompts:
        raise ValueError("At least one prompt is required")
    versions = {name: parameter._version for name, parameter in target.model.named_parameters()}
    rows = []
    for prompt in prompts:
        if prompt.ndim == 1:
            prompt = prompt.unsqueeze(0)
        if prompt.shape[0] != 1:
            raise ValueError("Each variable-length prompt entry must describe one request")
        prompt = prompt.to(target.device)
        prefill = target(prompt)
        serial_cache = target.new_cache()
        serial_logits, serial_states = [], []
        for token in prompt.split(1, dim=1):
            serial = target(token, serial_cache)
            serial_logits.append(serial.logits)
            serial_states.append(serial.round_hidden)
        prefill_comparison = _comparison(prefill.logits, torch.cat(serial_logits, dim=1), atol, rtol)
        hidden_comparison = _comparison(prefill.round_hidden, torch.cat(serial_states, dim=1), atol, rtol)
        native = target.model(
            input_ids=prompt, past_key_values=target.new_cache(), use_cache=True,
            exit_at_step=3, logits_to_keep=0,
        )
        native_comparison = _comparison(prefill.logits, native.logits, atol, rtol)

        continuation = greedy_generate(target, prompt, new_tokens)
        replay_cache = target.clone_cache(prefill.cache)
        replay_logits, replay_states = [], []
        for token in continuation.split(1, dim=1):
            replay_output = target(token, replay_cache)
            replay_logits.append(replay_output.logits)
            replay_states.append(replay_output.round_hidden)
        block_cache = target.clone_cache(prefill.cache)
        block_logits, block_states = [], []
        for block in continuation.split(block_size, dim=1):
            block_output = target(block, block_cache)
            block_logits.append(block_output.logits)
            block_states.append(block_output.round_hidden)
        block_comparison = _comparison(torch.cat(block_logits, 1), torch.cat(replay_logits, 1), atol, rtol)
        replay_hidden, block_hidden = torch.cat(replay_states, 1), torch.cat(block_states, 1)
        round_comparisons = [
            _comparison(block_hidden[:, :, round_index], replay_hidden[:, :, round_index], atol, rtol)
            for round_index in range(target.rounds)
        ]

        prefix_length = prompt.shape[1]
        target.crop_cache(block_cache, prefix_length)
        crop_output = target(continuation, block_cache)
        crop_comparison = _comparison(crop_output.logits, torch.cat(replay_logits, 1), atol, rtol)

        # A valid first token followed by a forced wrong token exercises both
        # accepted-prefix retention and rollback of later-round rejected KV.
        proposals = continuation[:, :min(block_size, new_tokens)].clone()
        forced_rejection = 1
        proposals[:, forced_rejection] = (proposals[:, forced_rejection] + 1) % target.vocab_size
        proposal_cache = target.clone_cache(prefill.cache)
        proposal_output = target(proposals, proposal_cache)
        predictions = torch.cat([prefill.logits[:, -1:], proposal_output.logits[:, :-1]], dim=1).argmax(-1)
        matches = proposals.eq(predictions)[0]
        mismatches = (~matches).nonzero(as_tuple=False)
        accepted = int(mismatches[0, 0]) if len(mismatches) else proposals.shape[1]
        if accepted == proposals.shape[1]:
            raise RuntimeError("A deliberately wrong proposal was unexpectedly accepted")
        target.crop_cache(proposal_cache, prefix_length + accepted)
        correction = predictions[:, accepted:accepted + 1]
        recovered = [proposals[:, :accepted], correction]
        state = target(correction, proposal_cache)
        for _ in range(new_tokens - accepted - 1):
            next_token = state.logits[:, -1].argmax(-1, keepdim=True)
            recovered.append(next_token)
            state = target(next_token, proposal_cache)
        recovered_tokens = torch.cat(recovered, dim=1)
        recovery_equal = bool(torch.equal(recovered_tokens, continuation))
        source_cache_unchanged = target.cache_lengths(prefill.cache) == [prefix_length] * target.logical_layers
        row = {
            "prompt_tokens": prefix_length, "continuation_tokens": continuation[0].tolist(),
            "round_hidden_shape": list(prefill.round_hidden.shape),
            "cache_slots": target.logical_layers, "prefill": prefill_comparison,
            "round_hidden_prefill": hidden_comparison, "native_final_round": native_comparison,
            "serial_vs_block": block_comparison, "crop_replay": crop_comparison,
            "round_hidden_serial_vs_block": round_comparisons,
            "forced_rejection_index": forced_rejection, "accepted_before_rejection": accepted,
            "recovery_tokens_equal": recovery_equal, "clone_source_unchanged": source_cache_unchanged,
        }
        comparisons = (prefill_comparison, native_comparison, block_comparison, crop_comparison)
        row["passed"] = (
            all(item["close"] and item["argmax_equal"] for item in comparisons)
            and hidden_comparison["close"] and accepted == forced_rejection
            and recovery_equal and source_cache_unchanged
        )
        rows.append(row)
    versions_unchanged = all(parameter._version == versions[name] for name, parameter in target.model.named_parameters())
    frozen = all(not parameter.requires_grad for parameter in target.model.parameters())
    return {
        "passed": all(row["passed"] for row in rows) and versions_unchanged and frozen,
        "rows": rows, "atol": atol, "rtol": rtol, "rounds": target.rounds,
        "hidden_semantics": "checkpoint hidden_states_list: shared RMSNorm after each round",
        "target_parameters_frozen": frozen, "parameter_versions_unchanged": versions_unchanged,
        "parameter_check": "PyTorch mutation version counters; not a full tensor checksum",
        "device": str(target.device), "attention_backend": target.model.config._attn_implementation,
        "dtype": str(next(target.model.parameters()).dtype),
    }


def main() -> None:
    """Run an explicitly requested small checkpoint validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attention-backend", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--memory-limit-gib", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.memory_limit_gib is not None and args.device.startswith("cuda"):
        device = torch.device(args.device)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(args.memory_limit_gib * 2**30 / total, 1.0), device)
    target = OuroTarget.from_pretrained(
        args.model, device=args.device, dtype=getattr(torch, args.dtype), attention_backend=args.attention_backend,
    )
    prompts = args.prompt or ["The capital of France is", "A sequence of integers: 1, 2, 3,"]
    ids = [target.tokenizer(prompt, return_tensors="pt").input_ids for prompt in prompts]
    report = validate_target(target, ids, new_tokens=args.new_tokens, block_size=args.block_size,
                             atol=args.atol, rtol=args.rtol)
    report["model"] = args.model
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
