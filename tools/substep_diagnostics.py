"""Detached pre-integration measurements; never refresh or modify simulation state."""

import numpy as np


class SubstepRecorder:
    def __init__(self):
        self.records = {}

    def record(self, **tensors):
        if self.records and set(tensors) != set(self.records):
            raise ValueError('substep fields changed during recording')
        for name, tensor in tensors.items():
            self.records.setdefault(name, []).append(tensor.detach().cpu().numpy().copy())

    def arrays(self):
        return {'substep_' + name: np.stack(rows) for name, rows in self.records.items()}
