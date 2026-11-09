"""Small CPU checks; no downloaded model or course model fixture."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from grpo_posttraining import core, sft, sft_analysis
from grpo_posttraining.artifacts import ROOT


class Tokenizer:
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return {'p': [1, 2], 'r': [3, 4], 's': [3], 'empty': []}[text].copy()


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 4)
        self.head = torch.nn.Linear(4, 5)
    def forward(self, ids):
        return SimpleNamespace(logits=self.head(self.embedding(ids)))


def test_collator_preserves_teacher_forcing_shift_and_response_only_mask():
    b = sft.collate_records([{'prompt': 'p', 'response': 'r'}, {'prompt': 'p', 'response': 's'}], Tokenizer())
    assert b['input_ids'].tolist() == [[1, 2, 3], [1, 2, 3]]
    assert b['labels'].tolist() == [[2, 3, 4], [2, 3, 0]]
    assert b['response_mask'].tolist() == [[False, True, True], [False, True, False]]


def test_empty_tokenized_prompt_is_rejected():
    with pytest.raises(ValueError):
        sft.collate_records([{'prompt': 'empty', 'response': 'r'}], Tokenizer())


@pytest.mark.parametrize('content', ['', '{"prompt":"p","response":""}', '{"prompt":2,"response":"r"}'])
def test_record_validation(tmp_path, content):
    p = tmp_path/'records.jsonl'; p.write_text(content)
    with pytest.raises(ValueError): sft.load_records(p)


def test_sft_accumulated_gradients_match_full_batch_and_mask_zeroes():
    mask = torch.tensor([[False, True, True], [False, True, False]])
    full = torch.full((2, 3), -2., requires_grad=True)
    micro = full.detach().clone().requires_grad_()
    core.run_sft_microbatch_train_step(full, mask, 1, 1.0)
    for i in range(2): core.run_sft_microbatch_train_step(micro[i:i+1], mask[i:i+1], 2, 1.0)
    assert torch.equal(full.grad, micro.grad)
    assert torch.equal(full.grad, -mask.float()/2)


@pytest.mark.parametrize('precision', sft.PRECISIONS)
def test_sft_training_loop_updates_tiny_cpu_model_once_per_accumulation(precision):
    torch.manual_seed(1)
    model = TinyLM()
    if precision == 'bf16': model = model.bfloat16()
    b = sft.collate_records([{'prompt': 'p', 'response': 'r'}], Tokenizer())
    before = model.head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
    rows = sft.train_loop(model, [b]*4, optimizer, device='cpu', precision=precision, grad_accum=2)
    assert len(rows) == 2
    assert rows[0]['response_tokens'] == 4
    assert rows[0]['input_tokens'] == 6
    assert rows[0]['peak_vram_gb'] is None
    assert not torch.equal(model.head.weight.detach(), before)
    assert optimizer.state[model.head.weight]['step'].item() == 2


def test_logged_nll_is_token_mean_but_objective_is_response_sum():
    torch.manual_seed(5)
    model = TinyLM()
    b = sft.collate_records([{'prompt': 'p', 'response': 'r'}], Tokenizer())
    with torch.no_grad():
        lp = core.run_get_response_log_probs(model, b['input_ids'], b['labels'], False)['log_probs']
        nll = -(lp*b['response_mask']).sum().item()
    stats = sft.microbatch(model, b, 4, 'bf16')
    assert stats['nll_sum'] == pytest.approx(nll)
    assert stats['loss'] == pytest.approx(nll/4)
    assert stats['response_tokens'] == 2


def test_training_requires_explicit_execution_flag_before_model_loading():
    with pytest.raises(ValueError, match='explicit'):
        sft.train(SimpleNamespace(execute_training=False))


def test_incomplete_accumulation_and_oversized_sequences_fail():
    model = TinyLM(); opt = torch.optim.AdamW(model.parameters())
    b = sft.collate_records([{'prompt': 'p', 'response': 'r'}], Tokenizer())
    with pytest.raises(RuntimeError, match='Incomplete'):
        sft.train_loop(model, [b], opt, device='cpu', precision='bf16', grad_accum=2)
    assert not opt.state
    with pytest.raises(ValueError, match='no truncation'):
        sft.train_loop(model, [b], opt, device='cpu', precision='bf16', max_seq_len=2)


def test_invalid_settings_rejected():
    with pytest.raises(ValueError): sft.validate_settings([1]*3, 1, 4, 2048, 1e-5, 'bf16')
    with pytest.raises(ValueError): sft.validate_settings([1]*4, 1, 4, 2048, 1e-5, 'fp16')


def test_archived_sft_metrics_and_paired_evaluation_recompute():
    r = sft_analysis.summarize()
    assert [r['evaluation'][a]['strict_correct'] for a in ['base', 'bf16', 'fp32']] == [0, 0, 3]
    assert [r['evaluation'][a]['format_pass'] for a in ['base', 'bf16', 'fp32']] == [8, 2, 5]
    assert [r['evaluation'][a]['at_token_budget'] for a in ['base', 'bf16', 'fp32']] == [7, 14, 44]
    assert all(r['training'][a]['optimizer_steps'] == 32 for a in ['bf16', 'fp32'])
    assert r['training_data']['records'] == 128
    assert r['training_data']['eos_targets'] == 0


def test_published_originals_match_archive_hashes():
    manifest = json.loads((ROOT/'provenance/original_selection.json').read_text(encoding='utf-8'))
    for item in manifest['published_originals']:
        assert hashlib.sha256((ROOT/item['published_path']).read_bytes()).hexdigest() == item['sha256']


def test_core_sft_bodies_match_published_original_adapters():
    original = ast.parse((ROOT/'original_implementation/adapters.py').read_text(encoding='utf-8'))
    maintained = ast.parse((ROOT/'src/grpo_posttraining/core.py').read_text(encoding='utf-8'))
    functions = lambda tree: {n.name: ast.dump(ast.Module(body=n.body, type_ignores=[]), include_attributes=False)
                              for n in tree.body if isinstance(n, ast.FunctionDef)}
    a, b = functions(original), functions(maintained)
    for name in ['run_tokenize_prompt_and_output', 'run_compute_entropy', 'run_get_response_log_probs',
                 'run_sft_microbatch_train_step', 'run_masked_normalize']:
        assert a[name] == b[name]
