#!/usr/bin/env python3
"""Read-only, CPU-only audit of saved CS336-A5 Base/GRPO MATH responses.

Strict-format rewards remain the official results. This helper identifies a
conservative set of final answers and reports how many strict failures might be
format/answer-extraction failures. It is not a standardized math benchmark.
"""
import json
import re
from collections import Counter
from pathlib import Path

from cs336_alignment.drgrpo_grader import extract_boxed_answer, grade

ROOT = Path('experiments/grpo_eval_step_0105')


def load(name):
    path = ROOT / f'{name}.jsonl'
    entries = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    data = {row['dataset_index']: row for row in entries}
    assert len(data) == len(entries), f'duplicate IDs in {path}'
    return data


def math_candidates(text):
    text = text.strip(' \n\t.:;')
    if not text:
        return []
    out = [text]
    if '\\boxed' in text:
        try:
            boxed = extract_boxed_answer(text)
            if boxed:
                out.append(boxed)
        except Exception:
            pass
    # Math regions at the end of an explicitly marked final answer.
    for pattern in (r'\\\((.*?)\\\)', r'\\\[(.*?)\\\]', r'\$([^$]+)\$'):
        matches = re.findall(pattern, text, flags=re.S)
        if matches:
            out.append(matches[-1].strip())
    # If prose explicitly concludes in a number or a TeX fraction, check that.
    for pattern in (
        r'(\\frac\s*\{[^}]+\}\s*\{[^}]+\})\s*[.。!]*$',
        r'(?<![\w.])(-?\d+(?:\.\d+)?(?:/\d+)?)\s*(?:years? old|years?|\.)?\s*$',
    ):
        match = re.search(pattern, text, flags=re.I)
        if match:
            out.append(match.group(1).strip())
    # Common multiple-choice decorations: (E) 10/3, etc.
    clean = re.sub(r'\\textbf\s*\{\s*\([A-E]\)\s*\}', '', text)
    clean = re.sub(r'^\s*\([A-E]\)\s*', '', clean)
    if clean != text:
        out.append(clean.strip())
    return list(dict.fromkeys(x for x in out if x))


def final_answer_candidates(response):
    text = response.strip()
    if '<answer>' in text:
        section = text.rsplit('<answer>', 1)[1].split('</answer>', 1)[0]
        return 'answer_tag', math_candidates(section)
    # Explicit boxed answer, last occurrence as in the course evaluator.
    if '\\boxed' in text:
        return 'boxed', math_candidates(text[text.rfind('\\boxed'):])
    # Explicit final answer cue; never guess from arbitrary intermediate numbers.
    cues = list(re.finditer(r'(?i)(?:^|\n|[.!?]\s+)(?:final\s+)?answer\s*[:：]|the\s+(?:final\s+)?answer\s+is\s+', text))
    if cues:
        return 'answer_cue', math_candidates(text[cues[-1].end():][-250:])
    return 'unresolved', []


def audit(data):
    counts = Counter()
    cases = []
    for idx, row in data.items():
        strict = row['scores']['reward'] == 1
        fmt = row['scores']['format_reward'] == 1
        counts['strict_correct'] += strict
        counts['strict_format_valid'] += fmt
        label, candidates = final_answer_candidates(row['response'])
        counts['mode_' + label] += 1
        possible = False
        matched = None
        for candidate in candidates:
            try:
                if grade(candidate, row['ground_truth']):
                    possible = True
                    matched = candidate
                    break
            except Exception:
                continue
        if strict or possible:
            counts['strict_or_extracted_correct'] += 1
        if not strict and possible:
            counts['newly_recovered'] += 1
            counts['recovered_from_' + label] += 1
        if not strict and not fmt and possible:
            counts['recovered_format_invalid'] += 1
        if not strict and fmt and possible:
            counts['recovered_format_valid'] += 1
        cases.append({'dataset_index': idx, 'strict': strict, 'format_valid': fmt,
                      'extraction_mode': label, 'candidate': matched,
                      'extracted_match': possible,
                      'ground_truth': row['ground_truth'],
                      'response_end': row['response'][-600:]})
    return counts, cases


def main():
    base, grpo = load('base'), load('grpo')
    assert len(base) == len(grpo) == 1024 and base.keys() == grpo.keys()
    n = len(base)
    bc, bcases = audit(base)
    gc, gcases = audit(grpo)
    gained = [i for i in base if not base[i]['scores']['reward'] and grpo[i]['scores']['reward']]
    format_invalid = sum(not bool(base[i]['scores']['format_reward']) for i in gained)
    print('Same 1024 questions. CPU-only re-grading; no model inference.\n')
    for name, counts in [('BASE', bc), ('GRPO', gc)]:
        print('===', name, '===')
        print(f"Strict correct: {counts['strict_correct']}/{n} ({100*counts['strict_correct']/n:.2f}%)")
        print(f"Strict format valid: {counts['strict_format_valid']}/{n} ({100*counts['strict_format_valid']/n:.2f}%)")
        print(f"Strict-wrong but explicit final answer matches GT: {counts['newly_recovered']}")
        print('  format invalid:', counts['recovered_format_invalid'],
              '| format valid:', counts['recovered_format_valid'])
        print(f"Strict correct OR explicit final answer match (audit estimate): {counts['strict_or_extracted_correct']}/{n} ({100*counts['strict_or_extracted_correct']/n:.2f}%)")
        print('  extraction modes:', {k.replace('mode_', ''): v for k,v in sorted(counts.items()) if k.startswith('mode_')})
    print('\n=== 601 STRICT REWARD GAINS: Base formatting split ===')
    print('Base format invalid:', format_invalid)
    print('Base format valid but answer/reward wrong:', len(gained)-format_invalid)
    report = {'n':n,'base':dict(bc),'grpo':dict(gc), 'strict_gains': len(gained),
              'strict_gains_base_format_invalid':format_invalid}
    out=ROOT/'format_free_audit.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    review=ROOT/'format_free_manual_review.jsonl'
    with review.open('w',encoding='utf-8') as f:
        for name, arr in [('base',bcases),('grpo',gcases)]:
            for row in arr:
                if not row['strict'] and row['extracted_match']:
                    f.write(json.dumps({'model': name,**row},ensure_ascii=False)+'\n')
    print('\nSaved:',out,review)
    print('IMPORTANT: Audit estimate is not an official accuracy; review recovered samples manually.')

if __name__=='__main__':
    main()
