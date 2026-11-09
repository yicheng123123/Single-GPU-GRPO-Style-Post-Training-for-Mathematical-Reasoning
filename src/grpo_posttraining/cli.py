from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from .artifacts import ROOT, new_run, sha256, model_inventory, write_json


def check_config(path):
    c=json.loads(Path(path).read_text())
    expected={'rollout_batch_size':256,'group_size':8,'gradient_accumulation_steps':128,
              'learning_rate':1e-5,'loss_type':'reinforce_with_baseline','normalize_by_std':True}
    for key,value in expected.items():
        if c.get(key)!=value:raise ValueError('Unsupported historical-baseline config: '+key)
    return {'configuration':'VALID_STATIC_CONFIGURATION','microbatch_responses':c['rollout_batch_size']//c['gradient_accumulation_steps'],
            'prompts_per_rollout':c['rollout_batch_size']//c['group_size'],
            'gpu_tested':False,'training_ready':'Requires local Base, train parquet and compatible ROCm runtime'}


def training(args):
    if not args.execute_training:raise ValueError('Training disabled: explicit --execute-training required')
    check_config(args.config)
    model=Path(args.model).resolve();data=Path(args.train_parquet).resolve()
    if not (model/'config.json').is_file() or not data.is_file():raise FileNotFoundError('Existing local model/data required')
    if any((model/n).exists() for n in ('run_config.json','train_metrics.jsonl','training_state.pt')):
        raise ValueError('--model is the original Base for a fresh run; use --resume for full training state')
    run=(ROOT/'.local/runs/training'/args.run_name).resolve()
    new_run(args.run_name,'training') if not args.resume else None
    if not run.is_relative_to((ROOT/'.local/runs/training').resolve()) or run==(ROOT/'.local/runs/training').resolve():raise ValueError('Unsafe training path')
    if args.resume:
        latest=json.loads((run/'latest_checkpoint.json').read_text())
        checkpoint=Path(latest['checkpoint']).resolve()
        if not checkpoint.is_relative_to(run) or not (checkpoint/'training_state.pt').is_file():
            raise FileNotFoundError('A full local optimizer/RNG checkpoint is required; archived Step 105 model alone is insufficient')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    if not torch.version.hip:raise RuntimeError('Existing ROCm environment required; no Windows GPU execution')
    identity={'base_files':model_inventory(model),'train_sha256':sha256(data),
              'config_sha256':sha256(args.config),
              'source_sha256':{p.name:sha256(p) for p in (ROOT/'src/grpo_posttraining').glob('*.py')},
              'historical_base_identity':'UNVERIFIED; this authenticates current initialization only'}
    if args.resume:
        initial=json.loads((run/'initialization_manifest.json').read_text())
        if identity!=initial:raise ValueError('Base/data/config/source changed since this run began')
    else:
        run.mkdir(parents=True,exist_ok=False)
        write_json(run/'initialization_manifest.json',identity)
    from .trainer import _training_main
    c=json.loads(Path(args.config).read_text())
    ns=argparse.Namespace(model=str(model),train_parquet=str(data),prompt_template=str(ROOT/'src/grpo_posttraining/prompts/r1_zero.prompt'),
        run_dir=str(run),max_steps=args.max_steps,checkpoint_interval=5,rollout_batch_size=c['rollout_batch_size'],
        group_size=c['group_size'],grad_accum=c['gradient_accumulation_steps'],seed=c['seed'],resume=args.resume,
        gpu_memory_utilization=.40,max_model_len=4096)
    _training_main(ns)


def main():
    p=argparse.ArgumentParser(description='CPU evidence tools and explicitly gated GPU entrypoints')
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('check-config');c.add_argument('--config',default=str(ROOT/'configs/train_grpo.json'))
    sub.add_parser('doctor')
    c=sub.add_parser('report');c.add_argument('--output',required=True)
    c=sub.add_parser('figures');c.add_argument('--output',required=True)
    c=sub.add_parser('examples');c.add_argument('--output',required=True)
    c=sub.add_parser('audit-data');c.add_argument('--train',required=True);c.add_argument('--test',required=True);c.add_argument('--legacy-dir',required=True);c.add_argument('--output',required=True)
    c=sub.add_parser('evaluate');c.add_argument('--model',required=True);c.add_argument('--tokenizer',required=True)
    c.add_argument('--label',required=True);c.add_argument('--run-name',required=True);c.add_argument('--cases',default=str(ROOT/'results/records/cases.json'))
    c.add_argument('--temperature',type=float,default=0.0);c.add_argument('--max-tokens',type=int,default=1024);c.add_argument('--seed',type=int,default=2026);c.add_argument('--execute-gpu',action='store_true')
    c=sub.add_parser('compare-runs');c.add_argument('--left',required=True);c.add_argument('--right',required=True);c.add_argument('--output',required=True)
    c=sub.add_parser('train');c.add_argument('--model',required=True);c.add_argument('--train-parquet',required=True)
    c.add_argument('--config',default=str(ROOT/'configs/train_grpo.json'));c.add_argument('--run-name',required=True);c.add_argument('--max-steps',type=int,required=True)
    c.add_argument('--resume',action='store_true');c.add_argument('--execute-training',action='store_true')
    args=p.parse_args()
    try:
        if args.command=='check-config':result=check_config(args.config)
        elif args.command=='doctor':
            versions={}
            for name in ('torch','vllm','transformers','pytest','matplotlib','pyarrow'):
                try:versions[name]=importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:versions[name]='NOT_INSTALLED'
            result={'packages':versions,'gpu_probed':False,'scope':'Package inventory only; not a training compatibility certificate'}
        elif args.command=='evaluate':
            from .evaluation import run
            result={'output':str(run(args))}
        elif args.command=='compare-runs':
            from .evaluation import compare
            result=compare(args.left,args.right,args.output)
        elif args.command=='train':result=training(args)
        else:
            from . import analysis
            if args.command=='audit-data':result=analysis.audit_data(args.train,args.test,args.legacy_dir,args.output)
            else:result=getattr(analysis,args.command)(args.output)
        print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as error:
        p.exit(2,str(error)+'\n')

