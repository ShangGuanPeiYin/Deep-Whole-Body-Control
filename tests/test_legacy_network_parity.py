"""Use the read-only legacy network as an independent numerical oracle."""

import importlib.util
import ast
import os
from pathlib import Path

import pytest
import torch

from dwbc_isaaclab.tasks.widow_go1.agents import dwbc_ppo_config
from dwbc_rsl_rl.modules import ActorCritic


def matched_models():
    root = Path(os.environ.get('DWBC_LEGACY_ROOT', '/home/xxs/research/Deep-Whole-Body-Control'))
    path = root / 'rsl_rl/rsl_rl/modules/actor_critic.py'
    if not path.exists():
        pytest.skip('read-only legacy oracle unavailable')
    spec = importlib.util.spec_from_file_location('legacy_actor_oracle', path)
    legacy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_module)
    torch.manual_seed(381)
    old = legacy_module.ActorCritic(
        76, 76, 18, actor_hidden_dims=[128], critic_hidden_dims=[128],
        leg_control_head_hidden_dims=[128,128], arm_control_head_hidden_dims=[128,128],
        num_leg_actions=12, num_arm_actions=6, adaptive_arm_gains=False,
        adaptive_arm_gains_scale=10., num_priv=24, num_hist=10, num_prop=76,
        init_std=[[.8,1.,1.]*4+[1.]*6])
    new = ActorCritic(**dwbc_ppo_config()['policy'])
    pairs = [(old.actor.priv_encoder,new.actor.priv_encoder),
             (old.actor.history_encoder,new.actor.history_encoder),
             (old.actor.actor_backbone,new.actor.backbone),
             (old.actor.actor_leg_control_head,new.actor.leg_head),
             (old.actor.actor_arm_control_head,new.actor.arm_head),
             (old.critic.critic_backbone,new.critic_backbone),
             (old.critic.critic_leg_control_head,new.critic_leg_head),
             (old.critic.critic_arm_control_head,new.critic_arm_head)]
    with torch.no_grad():
        for reference,candidate in pairs:
            left,right=list(reference.parameters()),list(candidate.parameters())
            assert len(left)==len(right)
            for a,b in zip(left,right):
                assert a.shape==b.shape
                b.copy_(a)
    return old,new,pairs,root


def test_network_outputs_logprob_and_gradients_match_legacy():
    old,new,pairs,_ = matched_models()
    obs = torch.randn(8,860)
    for history in (False,True):
        torch.testing.assert_close(new.act_inference(obs,history),old.act_inference(obs,history),atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(new.evaluate(obs),old.evaluate(obs),atol=1e-7,rtol=1e-6)
    old.update_distribution(obs,False)
    new.update_distribution(obs)
    actions=torch.randn(8,18)
    torch.testing.assert_close(new.get_actions_log_prob(actions),old.get_actions_log_prob(actions),atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(new.entropy,old.entropy,atol=1e-7,rtol=1e-6)
    (old.act_inference(obs).square().sum()+old.evaluate(obs).square().sum()).backward()
    (new.act_inference(obs).square().sum()+new.evaluate(obs).square().sum()).backward()
    for reference,candidate in pairs:
        for a,b in zip(reference.parameters(),candidate.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-6)


def test_full_ppo_update_matches_legacy():
    from dwbc_rsl_rl.algorithms import PPO
    old,new,pairs,root = matched_models()
    namespace = {'torch':torch,'nn':torch.nn,'optim':torch.optim,'ActorCritic':type(old)}
    for relative,name in [('storage/rollout_storage.py','RolloutStorage'),('algorithms/ppo.py','PPO')]:
        path = root / 'rsl_rl/rsl_rl' / relative
        definition = next(node for node in ast.parse(path.read_text()).body
                          if isinstance(node,ast.ClassDef) and node.name==name)
        exec(compile(ast.Module(body=[definition],type_ignores=[]),str(path),'exec'),namespace)
    common = dict(torque_supervision=False,num_learning_epochs=1,num_mini_batches=1,
                  min_policy_std=[0.05]*18,mixing_schedule=[0.5,0,1])
    reference = namespace['PPO'](old,adaptive_arm_gains=False,
                                 priv_reg_coef_schedual=[0.1,0.1,0,1],**common)
    candidate = PPO(new,priv_reg_schedule=[0.1,0.1,0,1],**common)
    for algorithm in (reference,candidate):
        algorithm.init_storage(2,3,[860],[860],[18])
    generator = torch.Generator().manual_seed(73)
    for step in range(3):
        obs = torch.randn(2,860,generator=generator)
        leg,arm = torch.randn(2,generator=generator),torch.randn(2,generator=generator)
        done = torch.tensor([step==2,False])
        for algorithm in (reference,candidate):
            torch.manual_seed(step+10)
            algorithm.act(obs,obs)
            algorithm.process_env_step(leg,arm,done,{'time_outs':done})
    for algorithm in (reference,candidate):
        algorithm.compute_returns(obs)
    torch.testing.assert_close(candidate.storage.returns,reference.storage.returns,atol=1e-7,rtol=1e-6)
    torch.manual_seed(92)
    expected = reference.update()
    torch.manual_seed(92)
    actual = candidate.update()
    for index,key in [(0,'value_loss'),(1,'surrogate_loss'),(2,'torque_loss'),(5,'priv_reg_loss')]:
        assert actual[key] == pytest.approx(expected[index],abs=1e-7,rel=1e-6)
    for left,right in pairs:
        for a,b in zip(left.parameters(),right.parameters()):
            torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
