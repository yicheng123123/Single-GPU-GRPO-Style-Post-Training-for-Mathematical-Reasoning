"""Byte identity and append-only output helpers; no model imports."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def jsonl(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    ids = [(r.get('model'),r['dataset_index']) for r in rows] if rows and 'dataset_index' in rows[0] else None
    if ids is not None and len(set(ids)) != len(ids):
        raise ValueError('Duplicate dataset_index: ' + str(path))
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def new_run(name, category='evaluations'):
    if not name or name in ('.', '..') or Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('Use a unique single-directory run name')
    directory = (ROOT / '.local/runs' / category / name).resolve()
    if not directory.is_relative_to((ROOT / '.local/runs').resolve()):
        raise ValueError('Output escapes local runs')
    if directory.exists():
        raise FileExistsError('Never reuse a result directory: ' + str(directory))
    return directory


def model_inventory(directory):
    directory = Path(directory).resolve()
    if not (directory / 'config.json').is_file():
        raise FileNotFoundError('Existing HF model config required')
    files = sorted(p for p in directory.rglob('*') if p.is_file())
    weights = [p for p in files if p.suffix in ('.safetensors', '.bin')]
    if not weights:
        raise FileNotFoundError('Existing local model weights required; no downloads')
    for index in directory.glob('*.index.json'):
        data = json.loads(index.read_text())
        for shard in set(data.get('weight_map', {}).values()):
            candidate = (directory / shard).resolve()
            if not candidate.is_relative_to(directory) or not candidate.is_file():
                raise ValueError('Missing or unsafe weight shard')
    return [{'path': p.relative_to(directory).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files]


def evaluation_manifest(model, cases, prompt, grader, settings):
    return {'model_files': model_inventory(model), 'cases_sha256': sha256(cases),
            'prompt_sha256': sha256(prompt), 'grader_sha256': sha256(grader),
            'settings': settings, 'historical_identity': 'No retroactive authentication of archived evaluation'}
