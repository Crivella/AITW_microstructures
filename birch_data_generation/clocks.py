"""Implement clock to time the execution of functions."""

import time
from functools import wraps
from typing import Dict

clocks: Dict[str, 'Clock'] = {}

from .loggers import logger


class Clock():
    def __new__(cls, name: str):
        if name in clocks:
            return clocks[name]
        new_clock = super().__new__(cls)
        clocks[name] = new_clock
        return new_clock

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.cumul = 0
        self.num_calls = 0

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            self.num_calls += 1
            self.cumul += time.time() - start
            return result

        return wrapper

    def report(self):
        if self.num_calls == 0:
            return
        tot_time = self.cumul
        avg_time = tot_time / self.num_calls * 1000

        logger.info(
            f'{self.name:>20s}   ({self.num_calls:>7d} CALLs): {tot_time:>13.4f} s  ({avg_time:>10.1f} ms/CALL)'
            )

    @staticmethod
    def report_all():
        total = Clock('total')
        for clock in clocks.values():
            clock.report()
            total.cumul += clock.cumul
            total.num_calls += clock.num_calls
