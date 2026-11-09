"""Fresh-only, manifest-bound evaluation. GPU execution is never implicit."""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path

from .artifacts import ROOT, evaluation_manifest, jsonl, new_run, sha256, write_json, model_inventory


def settings(temperature=0.0, max_tokens=1024, seed=2026):
    if temperature < 0 or max_tokens < 4:
        raise ValueError('Invalid sampling settings')
    return {'n':1,'temperature':temperature,'top_p':1.0,'min_tokens':4,
            'max_tokens':max_tokens,'stop':['</answer>'],'include_stop_str_in_output':True,'seed':seed}


def prepare(args):
    output=new_run(args.run_name)
    cases=Path(args.cases).resolve(); tokenizer=Path(args.tokenizer).resolve()
    if not cases.is_file() or not tokenizer.is_dir(): raise FileNotFoundError('Local cases and tokenizer required')
    data=json.loads(cases.read_text())['cases']
    if not data or len({r['dataset_index'] for r in data})!=len(data): raise ValueError('Empty/duplicate cases')
    prompt=ROOT/'src/grpo_posttraining/prompts/r1_zero.prompt'
    grader=ROOT/'src/grpo_posttraining/grader.py'
    policy_settings=settings(args.temperature,args.max_tokens,args.seed)
    manifest=evaluation_manifest(args.model,cases,prompt,grader,policy_settings)
    token_files=[p for p in tokenizer.iterdir() if p.is_file() and (p.name.startswith('tokenizer') or p.name in ('merges.txt','vocab.json','special_tokens_map.json','added_tokens.json'))]
    if not any(p.name=='tokenizer.json' for p in token_files): raise FileNotFoundError('Local tokenizer.json required')
    manifest.update(tokenizer_files=[{'path':p.name,'sha256':sha256(p)} for p in sorted(token_files)],
                    dtype='bfloat16',max_model_len=4096,engine='vllm',label=args.label)
    return output,manifest,data,prompt


def run(args):
    if not args.execute_gpu:
        raise ValueError('GPU generation disabled: --execute-gpu must be explicitly supplied')
    output,manifest,cases,prompt=prepare(args)
    import os
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    if not torch.version.hip:
        raise RuntimeError('Use the existing ROCm environment; Windows CPU preparation is not a GPU reproduction')
    from vllm import LLM, SamplingParams
    from .grader import r1_zero_reward_fn
    manifest['environment']={n:importlib.metadata.version(n) for n in ('torch','transformers','vllm')}
    manifest['torch_hip']=torch.version.hip
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'manifest.json',manifest)
    engine=LLM(model=str(Path(args.model).resolve()),tokenizer=str(Path(args.tokenizer).resolve()),
               dtype='bfloat16',tensor_parallel_size=1,gpu_memory_utilization=.40,
               max_model_len=4096,enforce_eager=True,seed=args.seed)
    template=prompt.read_text()
    generated=engine.generate([template.format(question=r['problem']) for r in cases],SamplingParams(**manifest['settings']))
    if len(generated)!=len(cases): raise RuntimeError('Incomplete generation')
    rows=[]
    with (output/'generations.jsonl').open('x',encoding='utf-8') as handle:
        for case,request in zip(cases,generated):
            completion=request.outputs[0]
            row={**case,'response':completion.text,'raw_generated_token_ids':list(completion.token_ids),
                 'generated_tokens':len(completion.token_ids),
                 'finish_reason':getattr(completion,'finish_reason',None) or 'UNKNOWN',
                 'stop_reason':getattr(completion,'stop_reason',None),
                 'scores':r1_zero_reward_fn(completion.text,case['ground_truth'],fast=True)}
            handle.write(json.dumps(row,ensure_ascii=False)+'\n');handle.flush();rows.append(row)
    if model_inventory(args.model)!=manifest['model_files']:
        raise RuntimeError('Model files changed during evaluation; outputs must not be published as a valid run')
    write_json(output/'metrics.json',{'n':len(rows),'strict_reward':sum(r['scores']['reward'] for r in rows)/len(rows),
                                    'format_rate':sum(r['scores']['format_reward'] for r in rows)/len(rows),
                                    'manifest_sha256':sha256(output/'manifest.json'),
                                    'generations_sha256':sha256(output/'generations.jsonl')})
    return output


def compare(left,right,output):
    left,right=Path(left),Path(right)
    manifests=[json.loads((p/'manifest.json').read_text()) for p in (left,right)]
    fields=('cases_sha256','prompt_sha256','grader_sha256','tokenizer_files','settings','dtype','max_model_len','engine','environment','torch_hip')
    for key in fields:
        if key not in manifests[0] or key not in manifests[1] or manifests[0][key]!=manifests[1][key]:
            raise ValueError('Cannot compare different evaluation protocols: '+key)
    rows=[jsonl(p/'generations.jsonl') for p in (left,right)]
    for directory,rs in zip((left,right),rows):
        bound=json.loads((directory/'metrics.json').read_text())
        if bound.get('manifest_sha256')!=sha256(directory/'manifest.json') or bound.get('generations_sha256')!=sha256(directory/'generations.jsonl') or bound.get('n')!=len(rs):
            raise ValueError('Results/manifest binding or completion check failed')
    maps=[{r['dataset_index']:r for r in rs} for rs in rows]
    if not maps[0] or maps[0].keys()!=maps[1].keys(): raise ValueError('Different case sets')
    for idx in maps[0]:
        if any(maps[0][idx][k]!=maps[1][idx][k] for k in ('problem','ground_truth')):raise ValueError('Mismatched cases')
    result={'n':len(maps[0]),'labels':[m['label'] for m in manifests],
            'strict_reward':[sum(r['scores']['reward'] for r in m.values())/len(m) for m in maps],
            'manifest_sha256':[sha256(p/'manifest.json') for p in (left,right)]}
    write_json(output,result)
    return result
