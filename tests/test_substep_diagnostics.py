import numpy as np
import pytest
import torch


def test_substep_recorder_copies_live_state_and_preserves_substep_order():
    from tools.substep_diagnostics import SubstepRecorder

    recorder = SubstepRecorder()
    live = torch.tensor([[1., 2.]])
    recorder.record(position=live, torque=live * 3)
    live.add_(4)
    recorder.record(position=live, torque=live * 3)
    arrays = recorder.arrays()
    np.testing.assert_array_equal(arrays['substep_position'], [[[1., 2.]], [[5., 6.]]])
    np.testing.assert_array_equal(arrays['substep_torque'], [[[3., 6.]], [[15., 18.]]])


def test_substep_recorder_rejects_changed_fields():
    from tools.substep_diagnostics import SubstepRecorder

    recorder = SubstepRecorder()
    recorder.record(position=torch.zeros(1, 20))
    with pytest.raises(ValueError, match='fields'):
        recorder.record(velocity=torch.zeros(1, 20))
