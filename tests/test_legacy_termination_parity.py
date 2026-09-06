"""Execute the unchanged default legacy termination rule on shared states."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dwbc_isaaclab.tasks.widow_go1.rewards import termination_flags


def test_default_termination_matches_legacy_including_timeout_boundary():
    root = Path(os.environ.get('DWBC_LEGACY_ROOT', '/home/xxs/research/Deep-Whole-Body-Control'))
    task = root / 'legged_gym/legged_gym/envs/widowGo1/widowGo1.py'
    math_source = root / 'legged_gym/legged_gym/utils/math.py'
    if not task.exists() or not math_source.exists():
        pytest.skip('legacy termination oracle unavailable')
    namespace = {'torch': torch}
    for path, name in ((math_source, 'euler_from_quat'), (task, 'check_termination')):
        function = next(node for node in ast.walk(ast.parse(path.read_text()))
                        if isinstance(node, ast.FunctionDef) and node.name == name)
        function.decorator_list = []
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    generator = torch.Generator().manual_seed(917)
    n = 1024
    quat = torch.randn(n, 4, generator=generator)
    quat /= quat.norm(dim=-1, keepdim=True)
    quat[:8] = torch.tensor([0., 0., 0., 1.])
    goal = torch.randn(n, 3, generator=generator)
    goal[:8] = 0
    height = .325 + .2 * torch.randn(n, generator=generator)
    height[:8] = torch.tensor([.325, .324, .326, .325, .325, .324, .326, .325])
    root_state = torch.zeros(n, 13)
    root_state[:, 2] = height
    lengths = torch.arange(n) % 502
    env = SimpleNamespace(
        contact_forces=torch.zeros(n, 27, 3),
        termination_contact_indices=torch.empty(0, dtype=torch.long),
        base_quat=quat, root_states=root_state, curr_ee_goal=goal,
        cfg=SimpleNamespace(termination=SimpleNamespace(z_threshold=.325)),
        episode_length_buf=lengths, max_episode_length=500,
    )
    namespace['check_termination'](env)
    roll, pitch, _ = namespace['euler_from_quat'](quat)
    terminated, reason = termination_flags(roll, pitch, height, goal)
    time_out = lengths > 500
    torch.testing.assert_close(terminated | time_out, env.reset_buf, atol=0, rtol=0)
    torch.testing.assert_close(time_out, env.time_out_buf, atol=0, rtol=0)
    assert not time_out[500] and time_out[501]
    assert not terminated[0] and terminated[1] and not terminated[2]
    torch.testing.assert_close(reason != 0, terminated, atol=0, rtol=0)
