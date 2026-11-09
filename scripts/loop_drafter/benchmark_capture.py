"""Paired capture microbenchmark on a dedicated/paused capture GPU; writes no training data."""
import argparse
from collections import defaultdict
import json,time
from pathlib import Path
import torch
from .target import OuroTarget
from .capture_protocol import capture_batch, _atomic_json
from .capture_fast import capture_batch_fast, capture_batch_graph


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--tasks',required=True)
    p.add_argument('--graph-only',action='store_true');p.add_argument('--production-check',action='store_true');p.add_argument('--out',type=Path,required=True);p.add_argument('--tokens',type=int,default=64)
    args=p.parse_args();torch.set_num_threads(4)
    target=OuroTarget.from_pretrained(args.model,device='cuda',dtype=torch.float32,attention_backend='sdpa')
    buckets=defaultdict(list)
    for line in Path(args.tasks).open():
        row=json.loads(line)['record'];buckets[len(row['input_ids'])].append(row)
    settings=[(100,4),(1000,2),(1800,1)] if args.production_check else [(100,4),(500,4)]
    choices=[(min((n for n,rows in buckets.items() if len(rows)>=b),key=lambda n:abs(n-x)),b) for x,b in settings]
    eos_id=target.tokenizer.eos_token_id
    eos=set(eos_id if isinstance(eos_id,list) else [eos_id]) if args.production_check else set()
    report=[]
    for length,batch in choices:
        ids=torch.tensor([r['input_ids'] for r in buckets[length][:batch]],device='cuda')
        capture_batch(target,ids,4,set())
        reference=None
        for name,fn in ([('baseline',capture_batch),('graph',capture_batch_graph)] if args.graph_only else [('baseline',capture_batch),('buffered',capture_batch_fast),
                        ('buffered_no_kv',lambda *a:capture_batch_fast(*a,buffered_cache=False))]):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            output=fn(target,ids,args.tokens,eos)
            torch.cuda.synchronize();duration=time.perf_counter()-start
            if reference is None: reference=output
            same=all(torch.equal(a[0],b[0]) for a,b in zip(reference,output))
            error=max(float((a[1].float()-b[1].float()).abs().max()) if a[1].shape==b[1].shape else float('inf') for a,b in zip(reference,output))
            record=dict(method=name,prompt_length=length,batch=len(ids),tokens=args.tokens,
                        seconds=duration,token_identity=same,hidden_max_abs=error,generated_lengths=[len(x[0])-length for x in output],
                        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
            report.append(record); print(json.dumps(record),flush=True);_atomic_json(args.out,report)
    if args.graph_only: return
    # Attribute CPU/GPU work in a short reference capture.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        capture_batch(target,ids,16,set())
    args.out.with_suffix('.profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=25))

if __name__=='__main__':main()
