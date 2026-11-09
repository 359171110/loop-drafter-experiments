"""Compare accumulation implementations without advancing production training."""
import argparse,json,time,statistics,fcntl
from pathlib import Path
import torch
from .loop_memory_draft import load_loop_memory_checkpoint
from .train_protocol import Settings,prepare_requests,epoch_order
from .train_loop_memory import accumulate_gradients
from .auto_train_when_ready import gpu_inventory,eligible_gpus


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(8)
    lease=args.run.parent/'gpu_leases'/f'gpu{args.gpu}.lock'
    with lease.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.gpu not in eligible_gpus(gpu_inventory()):raise RuntimeError('benchmark GPU occupied')
        reports=[]
        for mode in ['none','final','final_capacity','trajectory']:
            model,payload=load_loop_memory_checkpoint(args.run/mode/'latest.pt',device='cuda')
            torch.cuda.reset_peak_memory_stats()
            model.train();settings=Settings(**payload['settings']);cursor=payload['cursor']
            epoch=min(cursor['epoch'],settings.epochs-1)
            order=epoch_order(len(payload['data_index']),settings.seed,epoch)
            indices=order[cursor['next_request']:cursor['next_request']+settings.requests_per_update]
            requests=prepare_requests(payload['data_index'],indices,settings,epoch)
            reference=None;stats_reference=None;times={};identical=True;maxdiff=0.
            for backend,fn in [('reference',accumulate_gradients),('micro64',lambda model,requests,micro,device:accumulate_gradients(model,requests,64,device))]:
                torch.set_num_threads(8)
                timings=[]
                for repeat in range(3):
                    model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                    stats=fn(model,requests,settings.microbatch,torch.device('cuda'))
                    torch.cuda.synchronize();timings.append(time.perf_counter()-start)
                times[backend]=statistics.median(timings[1:])
                gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
                if reference is None:reference=gradients;stats_reference=stats
                else:
                    identical=reference.keys()==gradients.keys() and all(torch.equal(reference[n],g) for n,g in gradients.items())
                    maxdiff=max(float((reference[n]-g).abs().max()) for n,g in gradients.items())
            report=dict(mode=mode,checkpoint_step=cursor['step'],seconds=times,speedup=times['reference']/times['micro64'],
                        stats_identical=stats_reference==stats,gradients_bitwise_identical=identical,gradient_max_abs=maxdiff,loss_reference=stats_reference['loss'],loss_micro64=stats['loss'],
                        gradient_relative_l2=(sum(float((reference[n]-g).double().square().sum()) for n,g in gradients.items()) / max(1e-30,sum(float(g.double().square().sum()) for g in reference.values())))**.5,
                        peak_allocated=torch.cuda.max_memory_allocated())
            reports.append(report);args.out.write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(report),flush=True)
            del model,payload,reference,gradients,requests;torch.cuda.empty_cache()

if __name__=='__main__':main()
