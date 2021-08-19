import numpy as np


class LinearSchedule:
    def __init__(self, n_steps=100, start=0.0, end=-9.81, rounding=True):
        self.current = start
        self.step_size = (end - start) / n_steps
        self.n_steps = n_steps
        self.current_step = 0.0
        self.rounding = rounding

    def __call__(self):
        self.current_step += 1
        if self.current_step <= self.n_steps:
            self.current += self.step_size
        if self.rounding:
            self.current = round(self.current, 4)
        return self.current
