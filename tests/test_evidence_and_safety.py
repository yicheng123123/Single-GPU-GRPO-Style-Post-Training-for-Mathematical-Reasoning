import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from grpo_posttraining import analysis, evaluation
from grpo_posttraining.artifacts import ROOT, jsonl, model_inventory, new_run, sha256


def test_real_evidence_recomputed_matches_historical_cross_checks():
    r=analysis.summary()
    assert r['models']['base']['strict_correct']==148
    assert r['models']['grpo']['strict_correct']==743
    assert r['models']['base']['supplementary_audit_count']==437
    assert r['models']['grpo']['supplementary_audit_count']==744
    assert r['paired']['wrong_to_right']==601 and r['paired']['right_to_wrong']==6
    assert r['training']['logged_updates']==109
    assert r['evaluation_checkpoint_step_label']==105


def test_current_core_source_segments_match_imported_original_hashes():
    source=(ROOT/'src/grpo_posttraining/core.py').read_text(encoding='utf-8')
    nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    for item in json.loads((ROOT/'provenance/core_function_map.json').read_text()):
        assert hashlib.sha256(ast.get_source_segment(source,nodes[item['function']]).encode()).hexdigest()==item['source_segment_sha256']


def test_historical_records_have_no_duplicate_ids_or_case_mismatches():
    c,b,g=analysis.load_pairs()
    assert len(c)==len(b)==len(g)


def test_duplicate_records_rejected(tmp_path):
    p=tmp_path/'duplicate.jsonl';p.write_text('{"dataset_index":1}\n{"dataset_index":1}\n')
    with pytest.raises(ValueError):jsonl(p)


@pytest.mark.parametrize('name',['..','.','../archive','bad/name','bad\\name',''])
def test_run_paths_cannot_escape_or_use_dot_names(name):
    with pytest.raises(ValueError):new_run(name)


def test_evaluation_never_reuses_existing_output(monkeypatch,tmp_path):
    import grpo_posttraining.artifacts as artifacts
    monkeypatch.setattr(artifacts,'ROOT',tmp_path)
    p=tmp_path/'.local/runs/evaluations/existing';p.mkdir(parents=True)
    with pytest.raises(FileExistsError):new_run('existing')


def test_model_byte_changes_change_identity(tmp_path):
    (tmp_path/'config.json').write_text('{}');w=tmp_path/'model.safetensors';w.write_bytes(b'fixture-not-loaded')
    a=model_inventory(tmp_path);w.write_bytes(b'changed-fixture-not-loaded');b=model_inventory(tmp_path)
    assert a!=b


@pytest.mark.parametrize('field',['settings','cases_sha256','prompt_sha256','tokenizer_files','environment'])
def test_comparison_refuses_protocol_changes(tmp_path,field):
    manifest={k:{} for k in ('settings','cases_sha256','prompt_sha256','grader_sha256','tokenizer_files','environment','dtype','max_model_len','engine','torch_hip')}
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    (a/'manifest.json').write_text(json.dumps(manifest));manifest[field]='changed';(b/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):evaluation.compare(a,b,tmp_path/'should_not_exist.json')
    assert not (tmp_path/'should_not_exist.json').exists()


@pytest.mark.parametrize('command,args,message',[
    ('evaluate',['--model','missing','--tokenizer','missing','--label','test','--run-name','not-created'],'GPU generation disabled'),
    ('train',['--model','missing','--train-parquet','missing','--run-name','not-created','--max-steps','1'],'Training disabled')])
def test_gpu_entrypoints_require_explicit_opt_in(command,args,message):
    result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/project.py'),command]+args,
                          capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode==2 and message in result.stderr
    assert not (ROOT/'.local/runs/evaluations/not-created').exists()


def test_resume_from_model_only_archive_is_not_advertised_as_complete():
    m=json.loads((ROOT/'provenance/current_model_inventory.json').read_text())
    assert not m['exact_resume_from_this_archive']
    assert m['historical_evaluation_weight_identity'].startswith('UNVERIFIED')


def test_new_result_comparison_requires_content_binding(tmp_path):
    manifest={k:{} for k in ('settings','cases_sha256','prompt_sha256','grader_sha256','tokenizer_files','environment','dtype','max_model_len','engine','torch_hip')}
    row={'dataset_index':1,'problem':'toy','ground_truth':'1','response':'</think> <answer>1</answer>','scores':{'reward':1}}
    for name in ('a','b'):
        d=tmp_path/name;d.mkdir();(d/'manifest.json').write_text(json.dumps({**manifest,'label':name}))
        (d/'generations.jsonl').write_text(json.dumps(row)+'\n')
        (d/'metrics.json').write_text(json.dumps({'n':1,'manifest_sha256':sha256(d/'manifest.json'),'generations_sha256':sha256(d/'generations.jsonl')}))
    result=evaluation.compare(tmp_path/'a',tmp_path/'b',tmp_path/'valid.json')
    assert result['strict_reward']==[1.,1.]
    (tmp_path/'a/generations.jsonl').write_text(json.dumps({**row,'response':'altered'})+'\n')
    with pytest.raises(ValueError):evaluation.compare(tmp_path/'a',tmp_path/'b',tmp_path/'invalid.json')


def test_example_selection_uses_numeric_errors_for_strict_gains(tmp_path):
    rows=analysis.examples(tmp_path/'examples')
    gains=[r for r in rows if r['category']=='strict_gain']
    assert len(gains)==2
    assert {r['dataset_index'] for r in gains}=={177,193}
    assert len([r for r in rows if r['category']=='base_explicit_answer_format_failure'])==2
    with pytest.raises(FileExistsError):analysis.examples(tmp_path/'examples')
