#!/usr/bin/env python3
"""CS336 Spring 2025 A5: single-GPU on-policy GRPO (Qwen2.5-Math-1.5B).

The trainer stays on GPU 0. Each vLLM rollout is a short-lived subprocess on
that same GPU: it terminates *before* the trainer performs backward. The next
rollout explicitly loads the just-updated Hugging Face policy checkpoint.

Resume from a full, self-contained checkpoint containing model and AdamW state.
This is an independent implementation using the student's tested A5 adapters;
it is not a drop-in replacement for the instructor's original two-GPU runner.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
import hashlib


def _atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def _rollout_worker(payload_path: Path, output_path: Path):
    # Imported only inside worker, so the parent never starts a vLLM engine.
    from vllm import LLM, SamplingParams

    payload = json.loads(payload_path.read_text(encoding='utf-8'))
    llm = LLM(
        model=payload['model'],
        tensor_parallel_size=1,
        dtype='bfloat16',
        gpu_memory_utilization=payload['gpu_memory_utilization'],
        max_model_len=payload['max_model_len'],
        enforce_eager=True,  # safe default on ROCm; tune later if supported
        seed=payload['seed'],
    )
    sampling = SamplingParams(
        n=payload['group_size'],
        temperature=1.0,
        top_p=1.0,
        min_tokens=4,
        max_tokens=1024,
        stop=['</answer>'],
        include_stop_str_in_output=True,
        seed=payload['seed'],
    )
    outs = llm.generate(payload['prompts'], sampling, use_tqdm=True)
    if len(outs) != len(payload['prompts']):
        raise RuntimeError(f"Expected {len(payload['prompts'])} requests, got {len(outs)}")
    texts = []
    for i, req in enumerate(outs):
        group = [c.text for c in sorted(req.outputs, key=lambda x: x.index)]
        if len(group) != payload['group_size']:
            raise RuntimeError(f"Prompt {i}: expected {payload['group_size']} completions, got {len(group)}")
        texts.extend(group)
    _atomic_json(output_path, {
        'step': payload['step'],
        'dataset_indices': payload['dataset_indices'],
        'group_size': payload['group_size'],
        'responses': texts,
    })
    print(f"Saved {len(texts)} official rollouts to {output_path}", flush=True)


def _new_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation='sdpa',
    ).to('cuda:0')
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model


def _save_policy_for_rollout(model, tokenizer, target_dir: Path):
    staging = target_dir.with_name(target_dir.name + '_staging')
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    # Train in FP32 (master weights), but export BF16 for efficient vLLM
    # inference. BF16 master parameters with LR=1e-5 can silently round small
    # optimizer steps to zero; keeping master weights in FP32 avoids this.
    import torch
    inference_state = {
        k: (v.detach().to(device='cpu', dtype=torch.bfloat16)
            if torch.is_floating_point(v) else v.detach().cpu())
        for k, v in model.state_dict().items()
    }
    model.save_pretrained(staging, state_dict=inference_state, safe_serialization=True)
    del inference_state
    tokenizer.save_pretrained(staging)
    # Do not erase a live vLLM model; the rollout subprocess has already exited.
    if target_dir.exists():
        shutil.rmtree(target_dir)
    os.replace(staging, target_dir)


def _save_checkpoint(run_dir: Path, completed_steps: int, model, tokenizer, optimizer):
    import torch
    folder = run_dir / 'checkpoints' / f'step_{completed_steps:04d}'
    staging = folder.with_name(folder.name + '_staging')
    if staging.exists() or folder.exists():
        raise FileExistsError('Preserve existing checkpoint/staging directory: ' + str(folder))
    staging.mkdir(parents=True)
    model.save_pretrained(staging / 'model', safe_serialization=True)
    tokenizer.save_pretrained(staging / 'model')
    torch.save({
        'optimizer': optimizer.state_dict(),
        'completed_steps': completed_steps,
        'cpu_rng': torch.get_rng_state(),
        'gpu_rng': torch.cuda.get_rng_state(0),
    }, staging / 'training_state.pt')
    os.replace(staging, folder)
    _atomic_json(run_dir / 'latest_checkpoint.json', {'checkpoint': str(folder), 'completed_steps': completed_steps})
    # Maintained runner preserves all full checkpoints; budget disk space explicitly.
    print(f"Full resume checkpoint: {folder}", flush=True)


def _training_main(args):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from grpo_posttraining.grader import r1_zero_reward_fn, extract_boxed_answer
    from grpo_posttraining.core import (
        run_compute_group_normalized_rewards,
        run_tokenize_prompt_and_output,
        run_get_response_log_probs,
        run_grpo_microbatch_train_step,
    )

    if not torch.cuda.is_available():
        raise RuntimeError('No ROCm/HIP GPU visible to torch.')
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f'Expected one visible GPU; found {torch.cuda.device_count()}.')
    if args.rollout_batch_size % args.group_size:
        raise ValueError('rollout_batch_size must be divisible by group_size')
    if args.rollout_batch_size % args.grad_accum:
        raise ValueError('rollout_batch_size must be divisible by grad_accum')
    if args.rollout_batch_size != 256 or args.group_size != 8 or args.grad_accum != 128:
        raise ValueError('This first on-policy baseline intentionally keeps the 256/8/128 instructor defaults.')

    run_dir = Path(args.run_dir).resolve()
    allowed = Path(__file__).resolve().parents[2] / '.local/runs'
    if not run_dir.is_relative_to(allowed.resolve()) or run_dir == allowed.resolve():
        raise ValueError('Training outputs must be an individual project .local/runs directory')
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'rollouts').mkdir(exist_ok=True)
    template = Path(args.prompt_template).read_text(encoding='utf-8')
    # The ModelScope mirror contains both aggregate and per-category files.
    # ONLY its aggregate train parquet is allowed here; globbing all parquets
    # would mix train+test and duplicate every sample.
    train_parquet = Path(args.train_parquet).expanduser().resolve()
    if not train_parquet.is_file():
        raise FileNotFoundError(f'MATH train parquet not found: {train_parquet}')
    if train_parquet.name != 'train-00000-of-00001.parquet' or train_parquet.parent.name != 'data':
        raise ValueError('Use data/math_full/data/train-00000-of-00001.parquet (aggregate TRAIN only).')
    dataset = load_dataset('parquet', data_files={'train': str(train_parquet)}, split='train')
    if len(dataset) != 7500:
        raise RuntimeError(f'Expected 7500 MATH training examples, got {len(dataset)}; refusing to train.')
    required_columns = {'problem', 'solution'}
    if not required_columns.issubset(dataset.column_names):
        raise RuntimeError(f'Expected columns {required_columns}, got {dataset.column_names}')
    ground_truth_answers = [extract_boxed_answer(s) for s in dataset['solution']]
    invalid = [i for i, answer in enumerate(ground_truth_answers)
               if not isinstance(answer, str) or not answer.strip()]
    # A few source MATH solutions contain no parseable \boxed{...} answer.
    # Do NOT pass None/full reasoning into the reward function, and never use
    # test examples as substitutes. Sample only from valid TRAIN row indices.
    if len(invalid) > 10:
        raise RuntimeError(f'Too many MATH solutions without boxed answers: {len(invalid)}; indices={invalid[:20]}')
    invalid_set = set(invalid)
    eligible_indices = [i for i in range(len(dataset)) if i not in invalid_set]
    _atomic_json(run_dir / 'excluded_invalid_answers.json', {
        'reason': 'No parseable boxed final answer in training solution',
        'excluded_row_indices': invalid,
        'total_rows': len(dataset),
        'eligible_rows': len(eligible_indices),
    })
    print(f'[dataset] {len(dataset)} TRAIN rows, {len(eligible_indices)} eligible; '
          f'excluded {len(invalid)} missing boxed answers at {invalid}', flush=True)
    with train_parquet.open('rb') as source:
        train_sha256 = hashlib.file_digest(source, 'sha256').hexdigest()
    print(f'[dataset] MATH train: {len(dataset)} rows; file={train_parquet}; sha256={train_sha256}', flush=True)

    config_file = run_dir / 'train_config.json'
    requested_config = {
        'model': args.model, 'seed': args.seed,
        'rollout_batch_size': args.rollout_batch_size,
        'group_size': args.group_size, 'gradient_accumulation_steps': args.grad_accum,
        'prompt_template': str(Path(args.prompt_template).resolve()),
        'learning_rate': 1e-5,
        'loss_type': 'reinforce_with_baseline', 'normalize_by_std': True,
        'dataset': 'ModelScope MATH-lighteval aggregate train (7500 rows, filtered invalid answers)',
        'excluded_row_indices': invalid,
        'eligible_rows': len(eligible_indices),
        'train_parquet': str(train_parquet),
        'train_sha256': train_sha256,
    }
    if config_file.exists():
        prior = json.loads(config_file.read_text(encoding='utf-8'))
        if prior != requested_config:
            raise RuntimeError('Existing run config differs from this invocation; use a new --run-dir')
    else:
        _atomic_json(config_file, requested_config)
    latest = run_dir / 'latest_checkpoint.json'
    if args.resume:
        if not latest.exists():
            raise RuntimeError('--resume supplied but no complete checkpoint exists')
        ckpt = Path(json.loads(latest.read_text())['checkpoint'])
        completed = int(json.loads(latest.read_text())['completed_steps'])
        model_source = str(ckpt / 'model')
        print(f'Resuming from step {completed}: {ckpt}', flush=True)
    else:
        if latest.exists():
            raise RuntimeError(f'{run_dir} has a checkpoint; add --resume or choose a new --run-dir')
        completed = 0
        model_source = args.model

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = _new_model(model_source)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.0)
    if args.resume:
        # Any rollout at/after the checkpoint boundary may have been generated
        # by an uncheckpointed policy before interruption: never reuse it.
        recovery = run_dir / 'recovery' / f'from_{completed:04d}_{time.time_ns()}'
        recovery.mkdir(parents=True)
        for old_result in (run_dir / 'rollouts').glob('step_*_responses.json'):
            try:
                old_step = int(old_result.name.split('_')[1])
            except (ValueError, IndexError):
                continue
            if old_step >= completed:
                shutil.move(str(old_result), str(recovery / old_result.name))
                old_request = old_result.with_name(old_result.name.replace('_responses', '_request'))
                if old_request.exists():
                    shutil.move(str(old_request), str(recovery / old_request.name))
        trusted_state = torch.load(ckpt / 'training_state.pt', map_location='cpu', weights_only=False)
        if trusted_state['completed_steps'] != completed:
            raise RuntimeError('Checkpoint metadata disagreement')
        optimizer.load_state_dict(trusted_state['optimizer'])
        torch.set_rng_state(trusted_state['cpu_rng'])
        torch.cuda.set_rng_state(trusted_state['gpu_rng'], device=0)
        # Checkpointed policy takes precedence over any interrupted rollout staging files.
        _save_policy_for_rollout(policy, tokenizer, run_dir / 'rollout_policy')
        rollout_model = str(run_dir / 'rollout_policy')
        # A crash can leave metric lines newer than the last full checkpoint.
        # Those are no longer part of the resumed, reproducible training run.
        metrics = run_dir / 'metrics.jsonl'
        if metrics.exists():
            shutil.copy2(metrics, recovery / 'metrics_before_resume.jsonl')
            kept = []
            for line in metrics.read_text(encoding='utf-8').splitlines():
                item = json.loads(line)
                if item['completed_steps'] <= completed:
                    kept.append(line)
            metrics.write_text('\n'.join(kept) + ('\n' if kept else ''), encoding='utf-8')
    else:
        rollout_model = model_source

    n_prompts = args.rollout_batch_size // args.group_size
    for step in range(completed, args.max_steps):
        started = time.monotonic()
        # Deterministic prompt selection across recovery (without relying on global RNG).
        indices = random.Random(args.seed + step).sample(eligible_indices, n_prompts)
        prompts = [template.format(question=dataset[i]['problem']) for i in indices]
        ground_truths = [ground_truth_answers[i] for i in indices]
        request = run_dir / 'rollouts' / f'step_{step:04d}_request.json'
        result = run_dir / 'rollouts' / f'step_{step:04d}_responses.json'
        _atomic_json(request, {
            'model': rollout_model,
            'step': step,
            'seed': args.seed + step,
            'group_size': args.group_size,
            'dataset_indices': indices,
            'gpu_memory_utilization': args.gpu_memory_utilization,
            'max_model_len': args.max_model_len,
            'prompts': prompts,
        })
        torch.cuda.empty_cache()
        if not result.exists():
            print(f'[step {step}] Generating {args.rollout_batch_size} responses on GPU 0', flush=True)
            subprocess.run([sys.executable, '-m', 'grpo_posttraining.trainer', '_rollout',
                            '--payload', str(request), '--result', str(result)], check=True)
        responses_obj = json.loads(result.read_text(encoding='utf-8'))
        if (responses_obj['step'] != step or responses_obj['dataset_indices'] != indices
                or len(responses_obj['responses']) != args.rollout_batch_size):
            raise RuntimeError(f'Invalid / stale rollout record for step {step}; aborting')
        responses = responses_obj['responses']
        repeated_truths = [gt for gt in ground_truths for _ in range(args.group_size)]
        repeated_prompts = [p for p in prompts for _ in range(args.group_size)]
        advantages, raw_rewards, reward_meta = run_compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=responses,
            repeated_ground_truths=repeated_truths,
            group_size=args.group_size,
            advantage_eps=1e-6,
            normalize_by_std=True,
        )
        if advantages.numel() != args.rollout_batch_size:
            raise RuntimeError('Incorrect advantage count')
        batch = run_tokenize_prompt_and_output(repeated_prompts, responses, tokenizer)
        perm = torch.randperm(args.rollout_batch_size)
        for key in ('input_ids', 'labels', 'response_mask'):
            if key not in batch or batch[key].shape[0] != args.rollout_batch_size:
                raise RuntimeError(f'Invalid tokenization output {key}')
        mb_size = args.rollout_batch_size // args.grad_accum  # exactly 2
        optimizer.zero_grad(set_to_none=True)
        policy.train()
        losses = []
        for i in range(args.grad_accum):
            sl = perm[i * mb_size:(i + 1) * mb_size]
            inp = batch['input_ids'][sl].to('cuda:0', non_blocking=True)
            labels = batch['labels'][sl].to('cuda:0', non_blocking=True)
            mask = batch['response_mask'][sl].to('cuda:0', non_blocking=True)
            adv = advantages[sl].reshape(-1, 1).to('cuda:0')
            rew = raw_rewards[sl].reshape(-1, 1).to('cuda:0')
            if mask.sum().item() == 0:
                raise RuntimeError(f'Empty response mask at microbatch {i}')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                scored = run_get_response_log_probs(
                    model=policy, input_ids=inp, labels=labels, return_token_entropy=False,
                )
            loss, _meta = run_grpo_microbatch_train_step(
                policy_log_probs=scored['log_probs'].float(),
                response_mask=mask,
                gradient_accumulation_steps=args.grad_accum,
                loss_type='reinforce_with_baseline',
                raw_rewards=rew,
                advantages=adv,
                old_log_probs=None,
                cliprange=None,
            )
            if not torch.isfinite(loss.detach()).all():
                raise RuntimeError(f'Non-finite loss at step={step}, microbatch={i}')
            losses.append(float(loss.detach()))
            if (i + 1) % 32 == 0:
                print(f'[step {step + 1}/{args.max_steps}] backward {i + 1}/{args.grad_accum}', flush=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f'Non-finite grad norm at step={step}')
        # Check an individual parameter actually changed (e.g. if a batch's
        # rewards were constant, the GRPO advantages could all be zero).
        tracked_param = next(
            (p for name, p in policy.named_parameters()
             if 'self_attn.q_proj.weight' in name and p.requires_grad), None
        )
        if tracked_param is None:
            raise RuntimeError('Expected Qwen attention q_proj weights not found')
        before_update = tracked_param.detach().flatten()[:4096].float().clone()
        optimizer.step()
        update_abs_mean = (tracked_param.detach().flatten()[:4096].float() - before_update).abs().mean().item()
        optimizer.zero_grad(set_to_none=True)
        elapsed = time.monotonic() - started
        metric = {
            'completed_steps': step + 1,
            'rollout_size': len(responses),
            'n_prompts': n_prompts,
            'reward_mean': float(raw_rewards.float().mean().item()),
            'advantage_mean': float(advantages.float().mean().item()),
            'loss_sum': float(sum(losses)),
            'grad_norm_preclip': float(grad_norm),
            'parameter_update_abs_mean_sample': float(update_abs_mean),
            'format_reward_mean': reward_meta.get('format_reward_mean') if isinstance(reward_meta, dict) else None,
            'answer_reward_mean': reward_meta.get('answer_reward_mean') if isinstance(reward_meta, dict) else None,
            'elapsed_seconds': round(elapsed, 2),
        }
        with (run_dir / 'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(metric, ensure_ascii=False) + '\n')
        print('[train]', json.dumps(metric, ensure_ascii=False), flush=True)
        # Fresh policy must be written before next vLLM subprocess starts.
        _save_policy_for_rollout(policy, tokenizer, run_dir / 'rollout_policy')
        rollout_model = str(run_dir / 'rollout_policy')
        if (step + 1) % args.checkpoint_interval == 0 or step + 1 == args.max_steps:
            _save_checkpoint(run_dir, step + 1, policy, tokenizer, optimizer)
    print(f'Completed through {args.max_steps} actual optimizer updates.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', '_rollout'], nargs='?', default='train')
    parser.add_argument('--model', default='Qwen/Qwen2.5-Math-1.5B')
    parser.add_argument('--train-parquet', default='data/math_full/data/train-00000-of-00001.parquet')
    parser.add_argument('--prompt-template', default='cs336_alignment/prompts/r1_zero.prompt')
    parser.add_argument('--run-dir', default='experiments/grpo_single_gpu')
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--checkpoint-interval', type=int, default=5)
    parser.add_argument('--rollout-batch-size', type=int, default=256)
    parser.add_argument('--group-size', type=int, default=8)
    parser.add_argument('--grad-accum', type=int, default=128)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.40)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--payload', type=Path)
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    if args.mode == '_rollout':
        if args.payload is None or args.result is None:
            parser.error('_rollout needs --payload and --result')
        _rollout_worker(args.payload, args.result)
    else:
        raise RuntimeError('Use the guarded scripts/project.py train entrypoint')


if __name__ == '__main__':
    main()
