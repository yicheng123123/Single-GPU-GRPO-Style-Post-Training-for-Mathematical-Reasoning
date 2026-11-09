"""Maintained historical SFT path: teacher forcing, response sums, two precisions.

Source mapping: provenance/sft_source_map.json. This entry has CPU coverage;
the archived original, rather than this refactor, produced the historical runs.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time

import torch

from . import core
from .artifacts import ROOT, model_inventory, new_run, sha256, write_json

PRECISIONS = ('bf16', 'fp32_bf16_autocast')


def load_records(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    if not rows:
        raise ValueError('Empty SFT dataset')
    for i, row in enumerate(rows):
        if any(not isinstance(row.get(k), str) or not row[k] for k in ('prompt', 'response')):
            raise ValueError(f'Record {i}: nonempty prompt/response strings required')
    return rows


def collate_records(records, tokenizer):
    # Encode separately without extra special tokens, as in the historical adapter.
    for r in records:
        if any(not tokenizer.encode(r[k], add_special_tokens=False) for k in ('prompt', 'response')):
            raise ValueError('Prompt/response must each contain at least one token')
    return core.run_tokenize_prompt_and_output(
        [r['prompt'] for r in records], [r['response'] for r in records], tokenizer)


def validate_settings(records, epochs, grad_accum, max_seq_len, lr, precision):
    if precision not in PRECISIONS:
        raise ValueError('Unsupported precision')
    if min(epochs, grad_accum, max_seq_len) <= 0 or not 0 < lr < float('inf'):
        raise ValueError('Positive finite training settings required')
    if not records or len(records) % grad_accum:
        raise ValueError('Every epoch must contain complete gradient accumulation groups')


def microbatch(model, batch, grad_accum, precision):
    """Backward once; sum normalization is historical, token mean is logging only."""
    if precision not in PRECISIONS or grad_accum <= 0:
        raise ValueError('Invalid microbatch settings')
    mask = batch['response_mask']
    if not mask.any():
        raise ValueError('Empty response mask')
    context = (torch.autocast(device_type=batch['input_ids'].device.type, dtype=torch.bfloat16)
               if precision == 'fp32_bf16_autocast' else nullcontext())
    with context:
        result = core.run_get_response_log_probs(model, batch['input_ids'], batch['labels'], False)
    log_probs = result['log_probs']
    if precision == 'fp32_bf16_autocast':
        log_probs = log_probs.float()
    if not torch.isfinite(log_probs).all():
        raise RuntimeError('Non-finite log probabilities')
    with torch.no_grad():
        nll_sum = -(log_probs.detach() * mask).sum().item()
    loss, _ = core.run_sft_microbatch_train_step(log_probs, mask, grad_accum, 1.0)
    if not torch.isfinite(loss.detach()):
        raise RuntimeError('Non-finite SFT loss')
    return {'loss': loss.detach().item(), 'nll_sum': nll_sum,
            'response_tokens': int(mask.sum()), 'input_tokens': batch['input_ids'].numel()}


def train_loop(model, loader, optimizer, *, device, precision, epochs=1,
               grad_accum=4, max_seq_len=2048, max_updates=0, on_update=None):
    """One optimizer step per complete accumulation group; CPU toy-testable."""
    device = torch.device(device)
    if max_updates < 0 or grad_accum <= 0 or epochs <= 0:
        raise ValueError('Invalid update/accumulation/epoch count')
    gpu = device.type == 'cuda'
    model.train()
    optimizer.zero_grad(set_to_none=True)
    rows = []
    for epoch in range(epochs):
        sums = dict(loss=0., nll_sum=0., response_tokens=0, input_tokens=0)
        count = 0
        if gpu:
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        start = time.perf_counter()
        for raw_batch in loader:
            batch = {k: v.to(device) for k, v in raw_batch.items()}
            if batch['input_ids'].shape[1] > max_seq_len:
                raise ValueError('Sequence exceeds limit; no truncation is performed')
            stats = microbatch(model, batch, grad_accum, precision)
            for key in sums:
                sums[key] += stats[key]
            count += 1
            if count < grad_accum:
                continue
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if gpu:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            row = {'epoch': epoch + 1, 'global_step': len(rows) + 1,
                   'train_loss': sums['loss'], 'token_nll': sums['nll_sum'] / sums['response_tokens'],
                   'grad_norm_pre_clip': norm.item(), 'input_tokens': sums['input_tokens'],
                   'response_tokens': sums['response_tokens'], 'step_time_sec': elapsed,
                   'tokens_per_sec': sums['input_tokens'] / max(elapsed, 1e-9),
                   'lr': optimizer.param_groups[0]['lr'],
                   'peak_vram_gb': torch.cuda.max_memory_allocated() / 1024**3 if gpu else None,
                   'peak_reserved_gb': torch.cuda.max_memory_reserved() / 1024**3 if gpu else None}
            rows.append(row)
            if on_update:
                on_update(row)
            if max_updates and len(rows) >= max_updates:
                return rows
            count = 0
            sums = dict(loss=0., nll_sum=0., response_tokens=0, input_tokens=0)
            if gpu:
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            start = time.perf_counter()
        if count:
            raise RuntimeError('Incomplete gradient accumulation; no partial update was applied')
    return rows


def train(args):
    """Local-only, fresh-run GPU entry. No downloading, auto-eval or resume."""
    if not args.execute_training:
        raise ValueError('SFT disabled: explicit --execute-training required')
    records = load_records(args.data)
    validate_settings(records, args.epochs, args.grad_accum, args.max_seq_len, args.lr, args.precision)
    if args.max_updates < 0:
        raise ValueError('max-updates must be nonnegative')
    model_path = Path(args.model).resolve()
    if any((model_path / n).exists() for n in ('run_config.json', 'train_metrics.jsonl', 'training_state.pt')):
        raise ValueError('Fresh SFT starts from Base; do not pass an earlier trained run')
    # Historical Base identity remains a limitation; current files are bound below.
    run = new_run(args.run_name, 'sft')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if not torch.cuda.is_available():
        raise RuntimeError('Existing CUDA/ROCm GPU environment required')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader
    identity = {'base_files': model_inventory(model_path), 'data_sha256': sha256(args.data),
                'source_sha256': {n: sha256(ROOT/'src/grpo_posttraining'/n) for n in ('sft.py', 'core.py')},
                'historical_base_identity': 'Unverified; current manifest cannot authenticate historical runs'}
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Validate all lengths before allocating the model or creating a run.
    for r in records:
        batch = collate_records([r], tokenizer)
        if batch['input_ids'].shape[1] > args.max_seq_len:
            raise ValueError('Sequence exceeds limit; no truncation is performed')
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(records, batch_size=1, shuffle=True, generator=generator,
                        num_workers=0, collate_fn=lambda rows: collate_records(rows, tokenizer))
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype,
                attn_implementation='sdpa', local_files_only=True).to('cuda')
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    run.mkdir(parents=True, exist_ok=False)
    write_json(run/'initialization_manifest.json', identity)
    write_json(run/'run_config.json', {**vars(args), 'microbatch_size': 1,
                    'normalize_constant': 1.0, 'clip_norm': 1.0,
                    'optimizer_betas': [0.9, 0.999], 'weight_decay': 0.0,
                    'eos_added': False, 'resume': False})
    with (run/'train_metrics.jsonl').open('x', encoding='utf-8') as f:
        def log(row):
            f.write(json.dumps(row)+'\n'); f.flush()
        metrics = train_loop(model, loader, optimizer, device='cuda', precision=args.precision,
                epochs=args.epochs, grad_accum=args.grad_accum, max_seq_len=args.max_seq_len,
                max_updates=args.max_updates, on_update=log)
    model.config.use_cache = True
    model.save_pretrained(run/'model', safe_serialization=True)
    tokenizer.save_pretrained(run/'model')
    write_json(run/'completion.json', {'optimizer_steps': len(metrics), 'checkpoint': 'model',
                                     'optimizer_rng_saved': False})
    return {'output': str(run), 'optimizer_steps': len(metrics)}
