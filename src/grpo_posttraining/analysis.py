"""Recompute published statistics from preserved records; never generate responses."""
from __future__ import annotations
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re
from statistics import mean, median

from .artifacts import ROOT, jsonl, sha256, write_json


def load_pairs(records=None):
    records = Path(records or ROOT / 'results/records')
    cases = json.loads((records / 'cases.json').read_text())['cases']
    base, grpo = jsonl(records / 'base.jsonl'), jsonl(records / 'grpo.jsonl')
    maps = [{r['dataset_index']: r for r in rows} for rows in (cases, base, grpo)]
    if not cases or not (maps[0].keys() == maps[1].keys() == maps[2].keys()):
        raise ValueError('Case sets must be nonempty and identical')
    for idx, case in maps[0].items():
        for model in maps[1:]:
            row = model[idx]
            if row['problem'] != case['problem'] or row['ground_truth'] != case['ground_truth']:
                raise ValueError('Question/gold mismatch: ' + str(idx))
            scores = row['scores']
            if any(scores[k] not in (0, 1) for k in ('reward', 'answer_reward', 'format_reward')):
                raise ValueError('Unexpected score')
            fmt = '</think> <answer>' in row['response'] and '</answer>' in row['response']
            if bool(scores['format_reward']) != fmt or scores['reward'] > scores['format_reward']:
                raise ValueError('Stored format gate inconsistent with frozen grader')
    return cases, maps[1], maps[2]


def summary(records=None, metrics=None):
    records = Path(records or ROOT / 'results/records')
    metrics = Path(metrics or ROOT / 'results/metrics/train_metrics.jsonl')
    cases, base, grpo = load_pairs(records)
    n = len(cases)
    recovered = jsonl(records / 'automatic_recovery_candidates.jsonl')
    recover_ids = {arm: set() for arm in ('base', 'grpo')}
    recovery_keys = set()
    for r in recovered:
        key = (r['model'], r['dataset_index'])
        if key in recovery_keys or r['dataset_index'] not in base or r['model'] not in recover_ids:
            raise ValueError('Invalid recovery candidates')
        recovery_keys.add(key)
        if r['strict'] or not r['extracted_match']:
            raise ValueError('Recovery file must contain automatic strict-failure candidates only')
        recover_ids[r['model']].add(r['dataset_index'])
    model_stats = {}
    for arm, rows in (('base', base), ('grpo', grpo)):
        strict_ids = {i for i, r in rows.items() if r['scores']['reward']}
        if strict_ids & recover_ids[arm]:
            raise ValueError('Recovery overlaps original strict successes')
        lengths = [len(r['response']) for r in rows.values()]
        model_stats[arm] = {
            'strict_correct': len(strict_ids), 'strict_reward': len(strict_ids) / n,
            'format_valid': sum(r['scores']['format_reward'] for r in rows.values()),
            'format_rate': sum(r['scores']['format_reward'] for r in rows.values()) / n,
            'format_valid_strict_failure': sum(r['scores']['format_reward'] and not r['scores']['reward'] for r in rows.values()),
            'supplementary_audit_count': len(strict_ids | recover_ids[arm]),
            'supplementary_audit_rate': len(strict_ids | recover_ids[arm]) / n,
            'automatically_recovered': len(recover_ids[arm]),
            'missing_answer_close': sum('</answer>' not in r['response'] for r in rows.values()),
            'response_char_median': median(lengths),
            'token_cap_rate': 'UNKNOWN: historical token IDs/counts/finish reasons not saved',
        }
    gains = [i for i in base if not base[i]['scores']['reward'] and grpo[i]['scores']['reward']]
    losses = [i for i in base if base[i]['scores']['reward'] and not grpo[i]['scores']['reward']]
    rows = jsonl(metrics)
    steps = [r['completed_steps'] for r in rows]
    if len(set(steps)) != len(steps) or steps != list(range(1, max(steps) + 1)):
        raise ValueError('Missing/duplicate/out-of-order training steps')
    evaluated_step = json.loads((ROOT / 'results/metrics/historical_comparison.json').read_text())['step']
    to_checkpoint = [r for r in rows if r['completed_steps'] <= evaluated_step]
    result = {
        'n': n, 'evaluation_checkpoint_step_label': evaluated_step,
        'historical_evaluation_weight_identity': 'UNVERIFIED', 'models': model_stats,
        'strict_delta_pp': 100 * (model_stats['grpo']['strict_reward'] - model_stats['base']['strict_reward']),
        'supplementary_delta_pp_NOT_OFFICIAL': 100 * (model_stats['grpo']['supplementary_audit_rate'] - model_stats['base']['supplementary_audit_rate']),
        'paired': {'wrong_to_right': len(gains), 'right_to_wrong': len(losses),
                   'both_correct': sum(base[i]['scores']['reward'] and grpo[i]['scores']['reward'] for i in base),
                   'both_wrong': sum(not base[i]['scores']['reward'] and not grpo[i]['scores']['reward'] for i in base),
                   'gains_base_format_invalid': sum(not base[i]['scores']['format_reward'] for i in gains)},
        'training': {'logged_updates': max(steps), 'metric_rows': len(rows),
                     'responses_through_evaluated_step': sum(r['rollout_size'] for r in to_checkpoint),
                     'first10_mean_reward': mean(r['reward_mean'] for r in to_checkpoint[:10]),
                     'last10_mean_reward_through_evaluated_step': mean(r['reward_mean'] for r in to_checkpoint[-10:]),
                     'recorded_step_seconds_through_evaluated_step': sum(r['elapsed_seconds'] for r in to_checkpoint),
                     'step_timer_excludes': ['initial model load', 'post-update export', 'checkpoint save', 'interruptions'],
                     'grad_norm_max_through_evaluated_step': max(r['grad_norm_preclip'] for r in to_checkpoint),
                     'clipped_updates_through_evaluated_step': sum(r['grad_norm_preclip'] > 1 for r in to_checkpoint)},
        'sources': {p.relative_to(ROOT).as_posix(): sha256(p) for p in [metrics, records / 'base.jsonl', records / 'grpo.jsonl', records / 'cases.json', records / 'automatic_recovery_candidates.jsonl']},
        'scope': 'Stored historical strict scores reaggregated; automatic extraction estimates are not official math accuracy or human-reviewed scores',
    }
    return result


def report(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new report directory')
    result = summary()
    write_json(output / 'summary.json', result)
    text = '# Recomputed historical results\n\n'
    text += f"N={result['n']}; evaluated checkpoint label={result['evaluation_checkpoint_step_label']}; logged updates={result['training']['logged_updates']}.\n\n"
    for arm, s in result['models'].items():
        text += f"- {arm}: strict {s['strict_correct']}/{result['n']}; format {s['format_valid']}/{result['n']}; automatic audit estimate {s['supplementary_audit_count']}/{result['n']} (NOT official accuracy).\n"
    text += '\nHistorical checkpoint identity remains unverified. See docs/limitations.md.\n'
    (output / 'REPORT.md').write_text(text, encoding='utf-8')
    return result


def figures(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ['training_reward', 'strict_reward', 'supplementary_audit', 'format_compliance', 'error_transition']
    if any((output / (n + '.' + ext)).exists() for n in names for ext in ('svg', 'png')):
        raise FileExistsError('Do not overwrite figures; choose a new output directory')
    r = summary()
    rows = jsonl(ROOT / 'results/metrics/train_metrics.jsonl')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'figure.dpi': 130})
    def finish(fig, name, caption):
        fig.text(.02, .015, caption, fontsize=8, color='#555555', wrap=True)
        fig.tight_layout(rect=(0, .105, 1, 1))
        fig.savefig(output / (name + '.svg'))
        fig.savefig(output / (name + '.png'), dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.8, 3.7))
    x = [v['completed_steps'] for v in rows]; y = [v['reward_mean'] for v in rows]
    ax.plot(x, y, color='#9abbd5', linewidth=1, label='Per-step rollout reward')
    ax.plot(x, [mean(y[max(0, i-9):i+1]) for i in range(len(y))], color='#245b87', linewidth=2, label='Trailing mean (up to 10 steps)')
    ax.axvline(r['evaluation_checkpoint_step_label'], color='#555555', linestyle='--', label='Evaluated checkpoint label')
    ax.set(title='Training rollout reward — logged updates', xlabel='Optimizer updates (steps)', ylabel='Mean strict rollout reward (0–1)', ylim=(0, 1))
    ax.legend(loc='lower right', fontsize=8)
    finish(fig, 'training_reward', 'Source: results/metrics/train_metrics.jsonl, full logged run. Training batches change; this is not held-out accuracy.')
    for name, key, title in [('strict_reward','strict_reward','Strict R1-Zero reward — format gated'),
                             ('supplementary_audit','supplementary_audit_rate','Automatic answer-extraction audit — NOT official accuracy'),
                             ('format_compliance','format_rate','Strict format compliance')]:
        fig, ax = plt.subplots(figsize=(7.8, 3.6))
        values = [100*r['models'][arm][key] for arm in ('base','grpo')]
        ax.bar(['Base','GRPO-style Step '+str(r['evaluation_checkpoint_step_label'])], values, color=['#a4afb7','#245b87'], width=.5)
        for i, value in enumerate(values): ax.text(i, value+1.5, f'{value:.2f}%', ha='center')
        ax.set(title=title, xlabel='Policy', ylabel='Questions (%)', ylim=(0, 105))
        caption = 'Source: preserved Base/GRPO JSONL; same '+str(r['n'])+' questions. '
        caption += 'Post-hoc automated estimates; not fully human reviewed.' if key=='supplementary_audit_rate' else 'Historical identity/config limitations remain.'
        finish(fig, name, caption)
    fig, ax = plt.subplots(figsize=(7.8, 3.3))
    vals = [r['paired']['wrong_to_right'], r['paired']['right_to_wrong']]
    ax.barh(['Wrong → Right','Right → Wrong'], vals, color=['#245b87','#a4afb7'])
    for i, value in enumerate(vals):ax.text(value+3,i,str(value),va='center')
    ax.set(title='Paired strict-reward transitions', xlabel='Questions (count)', ylabel='Base → GRPO-style outcome', xlim=(0,max(vals)*1.15))
    finish(fig, 'error_transition', 'Source: paired dataset_index in Base/GRPO JSONL; strict reward includes format gating.')
    write_json(output / 'figure_sources.json', {'source_hashes': r['sources'], 'plots': names, 'smoothing': 'training only: trailing up to 10 steps', 'new_model_runs': 0})
    return names


def final_section(response):
    if '<answer>' in response:
        return response.rsplit('<answer>', 1)[1].split('</answer>', 1)[0].strip()
    match = re.search(r'(?:^|\n)Answer:\s*(.*)', response, re.S)
    return match.group(1).strip() if match else None


def examples(output):
    if Path(output).exists():raise FileExistsError('Use a new example directory')
    cases, base, grpo = load_pairs()
    recover = {(r['model'],r['dataset_index']):r for r in jsonl(ROOT / 'results/records/automatic_recovery_candidates.jsonl')}
    gained = [i for i in base if not base[i]['scores']['reward'] and base[i]['scores']['format_reward'] and grpo[i]['scores']['reward'] and ('base',i) not in recover]
    def scalar(text):
        if not text: return None
        text=text.strip().strip('$').strip()
        if not re.fullmatch(r'[+-]?\d+(?:\.\d+)?(?:/[+-]?\d+)?',text): return None
        try:return Fraction(text)
        except (ValueError,ZeroDivisionError):return None
    # Avoid calling an expression/extraction failure a mathematical improvement.
    scalar_gains=[i for i in gained if scalar(final_section(base[i]['response'])) is not None
                  and scalar(base[i]['ground_truth']) is not None
                  and scalar(final_section(base[i]['response']))!=scalar(base[i]['ground_truth'])]
    choices = [(i,'strict_gain') for i in scalar_gains[:2]] + [(i,'base_explicit_answer_format_failure') for i in (77,523)] + [(3155,'grpo_remaining_error')]
    items=[]
    for idx, category in choices:
        if category=='base_explicit_answer_format_failure':
            assert ('base',idx) in recover and not base[idx]['scores']['format_reward']
        items.append({'dataset_index':idx,'category':category,'problem':base[idx]['problem'],
                      'ground_truth':base[idx]['ground_truth'],
                      'base_final_section':final_section(base[idx]['response']) or recover.get(('base',idx),{}).get('candidate'),
                      'grpo_final_section':final_section(grpo[idx]['response']),
                      'base_scores':base[idx]['scores'],'grpo_scores':grpo[idx]['scores'],
                      'base_record_line': list(base).index(idx)+1, 'grpo_record_line':list(grpo).index(idx)+1,
                      'interpretation':'Selected qualitative case, not a random sample; final-answer correctness does not certify the full reasoning trace.'})
    if len(items)!=5:raise ValueError('Not enough verified example categories')
    write_json(Path(output)/'selected_cases.json',items)
    text='# 五个真实案例\n\n仅展示最终答案区域；长区域明确截短，不据此认定整段CoT正确。\n\n'
    explanations={'strict_gain':'严格Reward由0变为1；Base原本格式有效，仍需区分数学错误与评分器表示差异。',
                  'base_explicit_answer_format_failure':'Base有明确且匹配标准答案的最终答案，但严格格式失败；属于经过选择的诊断案例，不增加正式准确率。',
                  'grpo_remaining_error':'GRPO格式有效仍答错，说明格式学习没有消除数学错误。'}
    for row in items:
        text+=f"## dataset_index={row['dataset_index']}\n\n题目：{row['problem']}\n\nGround Truth：`{row['ground_truth']}`\n\n"
        for arm in ('base','grpo'):
            value=row[arm+'_final_section'] or '没有可识别的完整最终答案'
            if len(value)>400:value=value[:400]+' … [truncated display]'
            text+=f"- {arm} 最终区域：{value}\n- {arm} 原始评分：`{json.dumps(row[arm+'_scores'])}`\n"
        text+='\n'+explanations[row['category']]+f" 来源：results/records/{'base.jsonl'}:{row['base_record_line']}、grpo.jsonl:{row['grpo_record_line']}。\n\n"
    (Path(output)/'CASES_CN.md').write_text(text,encoding='utf-8')
    return items


def audit_data(train_path, test_path, legacy_dir, output):
    import pyarrow.parquet as pq
    from .answer_text import extract_boxed_answer
    train=pq.read_table(train_path).to_pylist(); test=pq.read_table(test_path).to_pylist()
    cases,_,_=load_pairs()
    if len(train)!=7500 or len(test)!=5000: raise ValueError('Unexpected split sizes')
    norm=lambda s:' '.join(s.split())
    train_exact={r['problem'] for r in train}; train_norm={norm(r['problem']) for r in train}
    invalid=[i for i,r in enumerate(train) if not extract_boxed_answer(r['solution'])]
    mismatches=[c['dataset_index'] for c in cases if test[c['dataset_index']]['problem']!=c['problem'] or extract_boxed_answer(test[c['dataset_index']]['solution'])!=c['ground_truth']]
    old_texts=set()
    for path in Path(legacy_dir).rglob('base.jsonl'):
        old_texts.update(norm(r['question']) for r in jsonl(path))
    duplicates=[{'test_index':i,'train_indices':[j for j,t in enumerate(train) if norm(t['problem'])==norm(r['problem'])]} for i,r in enumerate(test) if norm(r['problem']) in train_norm]
    result={'train_rows':len(train),'test_rows':len(test),'eligible_train_rows':len(train)-len(invalid),
            'invalid_train_indices':invalid,'test_case_mismatches':mismatches,
            'exact_full_test_duplicate_indices':[i for i,r in enumerate(test) if r['problem'] in train_exact],
            'whitespace_normalized_full_test_duplicates':duplicates,
            'eval_train_exact_overlap':[c['dataset_index'] for c in cases if c['problem'] in train_exact],
            'eval_train_normalized_overlap':[c['dataset_index'] for c in cases if norm(c['problem']) in train_norm],
            'eval_previous_development_overlap':[c['dataset_index'] for c in cases if norm(c['problem']) in old_texts],
            'sources':{'train_sha256':sha256(train_path),'test_sha256':sha256(test_path)},
            'scope':'Exact/whitespace equality only; no semantic or pretraining decontamination claim'}
    for key in ('type','level'):
        result['by_'+key]={}
        _,b,g=load_pairs()
        for c in cases:
            d=result['by_'+key].setdefault(test[c['dataset_index']][key],{'n':0,'base_strict':0,'grpo_strict':0})
            d['n']+=1;d['base_strict']+=b[c['dataset_index']]['scores']['reward'];d['grpo_strict']+=g[c['dataset_index']]['scores']['reward']
    write_json(output,result)
    return result
