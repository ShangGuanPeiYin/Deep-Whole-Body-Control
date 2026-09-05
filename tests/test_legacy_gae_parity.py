"""Execute the legacy storage class itself as an independent GAE oracle."""

import ast
from pathlib import Path
import os

import pytest
import torch

from dwbc_rsl_rl.storage import RolloutStorage


def test_gae_returns_advantages_match_legacy_with_terminals():
    root = Path(os.environ.get('DWBC_LEGACY_ROOT','/home/xxs/research/Deep-Whole-Body-Control'))
    path = root / 'rsl_rl/rsl_rl/storage/rollout_storage.py'
    if not path.exists():
        pytest.skip('legacy oracle unavailable')
    tree = ast.parse(path.read_text())
    # Execute the unchanged storage class without importing obsolete package dependencies.
    definition = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='RolloutStorage')
    module = ast.Module(body=[definition],type_ignores=[])
    namespace = {'torch':torch}
    exec(compile(module,str(path),'exec'),namespace)
    reference = namespace['RolloutStorage'](3,7,[860],[860],[18])
    candidate = RolloutStorage(3,7,[860],[860],[18])
    torch.manual_seed(5)
    rewards = torch.randn(7,3,2)
    values = torch.randn(7,3,2)
    dones = torch.rand(7,3,1) > 0.75
    last = torch.randn(3,2)
    for storage in (reference,candidate):
        storage.rewards.copy_(rewards)
        storage.values.copy_(values)
        storage.dones.copy_(dones)
        storage.compute_returns(last,0.99,0.95)
    torch.testing.assert_close(candidate.returns,reference.returns,atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(candidate.advantages,reference.advantages,atol=1e-7,rtol=1e-6)
