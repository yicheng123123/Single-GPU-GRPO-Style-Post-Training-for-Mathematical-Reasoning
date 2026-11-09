import torch
import pytest
from grpo_posttraining import core


class ToyTokenizer:
    pad_token_id=0
    def encode(self,text,add_special_tokens=False):
        return {'prompt':[1,2],'short':[3],'long':[4,5]}[text].copy()


def test_shift_mask_and_padding_align_to_response_labels():
    batch=core.run_tokenize_prompt_and_output(['prompt','prompt'],['short','long'],ToyTokenizer())
    assert batch['input_ids'].tolist()==[[1,2,3],[1,2,4]]
    assert batch['labels'].tolist()==[[2,3,0],[2,4,5]]
    assert batch['response_mask'].tolist()==[[False,True,False],[False,True,True]]


def test_entropy_matches_known_uniform_distribution_and_shape():
    h=core.run_compute_entropy(torch.zeros(2,3,4))
    assert h.shape==(2,3)
    assert torch.allclose(h,torch.full((2,3),torch.log(torch.tensor(4.)).item()))


class TinyModel:
    def __call__(self,ids):
        class Output:pass
        o=Output();o.logits=torch.zeros(*ids.shape,4);return o


def test_log_probs_gather_and_optional_entropy_without_real_model():
    o=core.run_get_response_log_probs(TinyModel(),torch.tensor([[1,2]]),torch.tensor([[2,3]]),True)
    assert torch.allclose(o['log_probs'],torch.full((1,2),-torch.log(torch.tensor(4.)).item()))
    assert o['token_entropy'].shape==(1,2)


@pytest.mark.parametrize('normalized',[False,True])
def test_group_baseline_zero_sum_and_constant_reward_groups(normalized):
    rewards={'a':0.,'b':1.,'c':1.,'d':1.}
    def reward(r,gt):return {'reward':rewards[r]}
    a,r,_=core.run_compute_group_normalized_rewards(reward,['a','b','c','d'],['x']*4,2,1e-6,normalized)
    assert torch.allclose(a.reshape(2,2).sum(1),torch.zeros(2),atol=1e-6)
    assert a[0]<0 and a[1]>0
    assert torch.equal(a[2:],torch.zeros(2))


def test_policy_gradient_sign():
    p=torch.tensor([[-2.,-3.],[-4.,-5.]],requires_grad=True)
    a=torch.tensor([[1.],[-1.]])
    loss=core.run_compute_naive_policy_gradient_loss(a,p).mean();loss.backward()
    assert torch.equal(p.grad,torch.tensor([[-.25,-.25],[.25,.25]]))


def test_length_normalization_ignores_prompt_and_padding_gradients():
    p=torch.full((2,4),-2.,requires_grad=True)
    mask=torch.tensor([[False,True,False,False],[False,True,True,True]])
    core.run_grpo_microbatch_train_step(p,mask,1,'reinforce_with_baseline',advantages=torch.tensor([[1.],[-1.]]))
    assert torch.allclose(p.grad,torch.tensor([[0.,-.5,0.,0.],[0.,1/6,1/6,1/6]]))


def test_microbatch_accumulation_matches_one_full_batch_gradient():
    mask=torch.tensor([[False,True,True],[False,True,False],[True,True,True],[True,False,False]])
    advantages=torch.tensor([[1.],[-1.],[.5],[-.5]])
    full=torch.full((4,3),-2.,requires_grad=True)
    micro=full.detach().clone().requires_grad_()
    core.run_grpo_microbatch_train_step(full,mask,1,'reinforce_with_baseline',advantages=advantages)
    for sl in (slice(0,2),slice(2,4)):
        core.run_grpo_microbatch_train_step(micro[sl],mask[sl],2,'reinforce_with_baseline',advantages=advantages[sl])
    assert torch.allclose(full.grad,micro.grad)


def test_zero_advantages_produce_zero_gradient():
    p=torch.ones(2,3,requires_grad=True)
    core.run_grpo_microbatch_train_step(p,torch.ones_like(p,dtype=torch.bool),1,'reinforce_with_baseline',advantages=torch.zeros(2,1))
    assert torch.equal(p.grad,torch.zeros_like(p))


def test_sft_sum_normalization_and_masked_gradients():
    p=torch.full((2,3),-2.,requires_grad=True);mask=torch.tensor([[False,True,False],[False,True,True]])
    loss,_=core.run_sft_microbatch_train_step(p,mask,2,1.)
    assert loss.item()==1.5
    assert torch.equal(p.grad,-mask.float()/4)


def test_clip_helper_direction_for_positive_and_negative_advantages():
    a=torch.tensor([[1.],[-1.]])
    loss,_=core.run_compute_grpo_clip_loss(a,torch.log(torch.tensor([[1.5],[.5]])),torch.zeros(2,1),.2)
    assert torch.allclose(loss,torch.tensor([[-1.2],[.8]]))


def test_unknown_loss_type_is_rejected():
    with pytest.raises(ValueError):core.run_compute_policy_gradient_loss(torch.zeros(1,1),'unknown',None,None,None,None)
