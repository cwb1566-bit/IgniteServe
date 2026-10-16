"""GPU correctness coverage for full-forward prefill/mixed composite graphs.

Forces partition admission, not performance evidence. Each real forward writes
KV only once; comparisons use separate identical generation requests.
"""
import argparse
from collections import Counter
import os
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', required=True)
p.add_argument('--gpu', type=int, required=True)
a = p.parse_args()
assert not subprocess.check_output(['nvidia-smi', f'--id={a.gpu}',
    '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), VLLM_ENABLE_STARTUP_OPTIMIZATIONS='1',
    VLLM_STARTUP_DAG_MODE='off', VLLM_ENABLE_V1_MULTIPROCESSING='0',
    VLLM_FORCE_FULL_CUDAGRAPH_PRECAPTURE='1', VLLM_ADAPTIVE_CUDAGRAPH_MIN_HITS='1000000',
    VLLM_ENABLE_CONCURRENT_PIECEWISE_CUDAGRAPH_CAPTURE='1',
)
import torch
from vllm import LLM, SamplingParams
from vllm.v1.worker.cudagraph_adaptation import GraphAdapter, plan_partitions, workload_key

base_execute = GraphAdapter.execute
selected_phase = None
counts = Counter()
runners = []

def phase(flags):
    return 'prefill' if all(flags) else 'mixed' if any(flags) else 'decode'

def plan(self, meta, q, s, f):
    if not runners:
        runners.append(self.runner)
    if phase(f) != selected_phase or not self.supported(meta):
        return ()
    table = self.runner.cuda_graph_manager.adapter.cost_table
    table._workloads['eager',sum(q),workload_key(q,s,f),1] = 1000.
    for n in (2,3):
        table._workloads['split_overhead',sum(q),workload_key(q,s,f),n] = 1.
    for lo in range(len(q)):
        for hi in range(lo+1,len(q)+1):
            key = workload_key(q[lo:hi],s[lo:hi],f[lo:hi])
            for size in self.ready_sizes():
                if size >= sum(q[lo:hi]):
                    table._workloads['graph',size,key,1] = 1.
    result = plan_partitions(self.runner.cuda_graph_manager.adapter,q,s,f,captured_sizes=self.ready_sizes())
    self._composite_parts = result
    return result

def execute(self, parts, meta, q, s, **kw):
    out = base_execute(self,parts,meta,q,s,**kw)
    if out is not None:
        counts[selected_phase] += 1
    return out

GraphAdapter.plan = plan
GraphAdapter.execute = execute
GraphAdapter.find_best_match = lambda self,n,*args,**kw: self._eager(n,'phase-validation')
llm = None
try:
    llm = LLM(model=a.model,dtype='float16',max_model_len=512,max_num_seqs=8,
        max_num_batched_tokens=64,enable_chunked_prefill=True,enable_prefix_caching=False,
        gpu_memory_utilization=.45,seed=123,
        compilation_config={'cudagraph_capture_sizes':[1,2,4]+list(range(8,65,8))})
    tok = llm.get_tokenizer()
    prompts = []
    for i,n in enumerate((6,6,6,12,14,36,36)):
        base = tok.encode(['Explain a computer.', 'The capital of France is',
                           'Explain parallel computing.'][i%3],add_special_tokens=False)
        prompts.append({'prompt_token_ids':(base*(n//len(base)+1))[:n]})
    params = SamplingParams(temperature=0,max_tokens=12,ignore_eos=True,logprobs=1)
    reference = llm.generate(prompts,params,use_tqdm=False)
    def tokens(out):
        return [x.outputs[0].token_ids for x in out]
    def probs(out):
        return [row[t].logprob for x in out for t,row in zip(
            x.outputs[0].token_ids,x.outputs[0].logprobs)]
    maximum = 0.
    for selected_phase in ('prefill','mixed'):
        r = runners[0]
        r.cuda_graph_manager.capture._queue.join()
        torch.cuda.synchronize()
        r.cuda_graph_manager.composites = None
        for _ in range(8):
            out = llm.generate(prompts,params,use_tqdm=False)
            assert tokens(out) == tokens(reference), selected_phase
            maximum = max(maximum,max(abs(x-y) for x,y in zip(probs(out),probs(reference))))
        r.cuda_graph_manager.capture._queue.join()
        assert counts[selected_phase] > 0, counts
        assert not any(p.failed for p in r.cuda_graph_manager.composites.plans.values())
    assert maximum < .05, maximum
    print('COMPOSITE_PHASES_VERIFIED',dict(counts), 'max_logprob_error',maximum,flush=True)
finally:
    if llm is not None:
        llm.llm_engine.engine_core.shutdown()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
