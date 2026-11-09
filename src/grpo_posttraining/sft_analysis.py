"""CPU-only statistics/figures from immutable historical SFT records."""
from __future__ import annotations
import json
from pathlib import Path

from .artifacts import ROOT, jsonl, sha256, write_json

SOURCE = ROOT / 'results/sft'


def summarize(source=SOURCE):
    source = Path(source)
    output = {'scope': 'Historical SFT-128; distinct from the 1024-question GRPO evaluation',
              'evaluation_protocol': {'n': 50, 'seed': 42, 'per_question_seed': '42 + dataset_index',
                'temperature': 1.0, 'top_p': 1.0, 'top_k': 0, 'max_new_tokens': 1024,
                'stop_strings': ['</answer>'], 'parameters': 'FP32', 'forward': 'BF16 autocast',
                'eos_observation': 'No raw IDs or finish reason archived; hitting cap is not proof of no EOS'},
              'training': {}, 'evaluation': {}, 'source_sha256': {}}
    for arm in ('bf16', 'fp32'):
        path = source/arm/'train_metrics.jsonl'
        rows = jsonl(path)
        if [r['global_step'] for r in rows] != list(range(1, 33)):
            raise ValueError('Expected the archived 32-step run')
        seconds = sum(r['step_time_sec'] for r in rows)
        tokens = sum(r['response_tokens'] for r in rows)
        weighted = lambda rs: sum(r['token_nll']*r['response_tokens'] for r in rs)/sum(r['response_tokens'] for r in rs)
        output['training'][arm] = {'optimizer_steps': len(rows), 'response_tokens': tokens,
            'input_tokens': sum(r['input_tokens'] for r in rows),
            'token_weighted_nll': weighted(rows),
            'first8_token_weighted_nll': weighted(rows[:8]), 'last8_token_weighted_nll': weighted(rows[-8:]),
            'first8_mean_token_nll': sum(r['token_nll'] for r in rows[:8])/8,
            'last8_mean_token_nll': sum(r['token_nll'] for r in rows[-8:])/8,
            'sum_step_time_sec': seconds,
            'input_tokens_per_logged_second': sum(r['input_tokens'] for r in rows)/seconds,
            'grad_norm_min': min(r['grad_norm_pre_clip'] for r in rows),
            'grad_norm_max': max(r['grad_norm_pre_clip'] for r in rows),
            'clipped_updates': sum(r['grad_norm_pre_clip'] > 1 for r in rows),
            'peak_allocated_gib': max(r['peak_vram_gb'] for r in rows),
            'peak_reserved_gib': max(r['peak_reserved_gb'] for r in rows),
            'timing_scope': 'Sum of synchronized per-update windows, excludes initial loading and final export',
            'nll_scope': 'Shuffled training batches, not a fixed validation curve'}
        output['source_sha256'][path.relative_to(ROOT).as_posix()] = sha256(path)
    indices = json.loads((source/'evaluation/eval_indices.json').read_text())
    reference = None
    for arm in ('base', 'bf16', 'fp32'):
        path = source/'evaluation'/f'{arm}.jsonl'; rows = jsonl(path)
        if len(rows) != 50 or [r['dataset_index'] for r in rows] != indices:
            raise ValueError('Evaluation set/order mismatch')
        questions = [(r['dataset_index'], r['question'], r['ground_truth']) for r in rows]
        if reference is None: reference = questions
        elif questions != reference: raise ValueError('Question/gold mismatch')
        if any(r['model'] != arm or r['reward'] not in (0, 1) or r['format_reward'] not in (0, 1) or r['reward'] > r['format_reward'] for r in rows):
            raise ValueError('Invalid archived scores')
        lengths = sorted(r['generated_tokens'] for r in rows)
        output['evaluation'][arm] = {'n': 50,
            'strict_correct': sum(r['reward'] for r in rows),
            'strict_reward_rate': sum(r['reward'] for r in rows)/50,
            'format_pass': sum(r['format_reward'] for r in rows),
            'format_rate': sum(r['format_reward'] for r in rows)/50,
            'at_token_budget': sum(r['generated_tokens'] >= 1024 for r in rows),
            'at_token_budget_rate': sum(r['generated_tokens'] >= 1024 for r in rows)/50,
            'length_min': lengths[0], 'length_median': (lengths[24]+lengths[25])/2,
            'length_mean': sum(lengths)/50, 'length_max': lengths[-1]}
        output['source_sha256'][path.relative_to(ROOT).as_posix()] = sha256(path)
    output['training_data'] = json.loads((source/'training_data_statistics.json').read_text())
    return output


def report(output):
    dest = Path(output).resolve()
    if dest.exists(): raise FileExistsError(dest)
    result = summarize()
    write_json(dest/'summary.json', result)
    return {'output': str(dest), 'training': result['training'], 'evaluation': result['evaluation']}


def figures(output):
    dest = Path(output).resolve()
    if dest.exists(): raise FileExistsError(dest)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = summarize()
    dest.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    for arm, color in [('bf16', '#d97706'), ('fp32', '#2563eb')]:
        rows = jsonl(SOURCE/arm/'train_metrics.jsonl')
        ax.plot([r['global_step'] for r in rows], [r['token_nll'] for r in rows],
                marker='.', color=color, label='BF16 parameters' if arm=='bf16' else 'FP32 parameters + BF16 autocast')
    ax.set(title='SFT-128: response-token training NLL', xlabel='Optimizer update', ylabel='Response-token mean NLL (nats)')
    ax.legend(); ax.grid(alpha=.2)
    fig.text(.08, .02, 'Source: results/sft/{bf16,fp32}/train_metrics.jsonl | 128 traces, 1 epoch, LR 1e-5\nShuffled training batches; not validation loss. Optimized loss is response-summed NLL.', fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, 1))
    for ext in ('svg', 'png'): fig.savefig(dest/f'sft_training_nll.{ext}', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0))
    labels = ['Base', 'BF16 SFT', 'FP32 SFT']
    for ax, key, title in zip(axes, ['strict_correct', 'format_pass', 'at_token_budget'], ['Strict correct', 'Format pass', 'At 1024-token budget']):
        values = [summary['evaluation'][arm][key] for arm in ('base', 'bf16', 'fp32')]
        ax.bar(labels, [v*2 for v in values], color=['#64748b', '#d97706', '#2563eb'])
        for i, v in enumerate(values): ax.text(i, v*2+2, f'{int(v)}/50', ha='center', fontsize=9)
        ax.set(title=title, ylabel='Share of 50 questions (%)', ylim=(0, 105))
        ax.tick_params(axis='x', labelrotation=15)
    fig.suptitle('SFT vs Base: fixed 50-question MATH development evaluation')
    fig.text(.06, .01, 'Source: results/sft/evaluation/*.jsonl | temperature 1.0, seed 42, max_new_tokens 1024\nStrict R1-Zero reward; stop string </answer>. Separate protocol from GRPO-1024.', fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, .94))
    for ext in ('svg', 'png'): fig.savefig(dest/f'sft_vs_base.{ext}', dpi=160)
    plt.close(fig)
    write_json(dest/'figure_sources.json', {'source_sha256': summary['source_sha256'],
               'figures': ['sft_training_nll.svg', 'sft_vs_base.svg'],
               'conditions': summary['evaluation_protocol'], 'gpu_used': False})
    return {'output': str(dest), 'figures': 2}
