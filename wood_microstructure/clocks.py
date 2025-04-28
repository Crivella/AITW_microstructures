"""Implement clock to time the execution of functions."""

import time
from functools import wraps
from typing import Dict

clocks: Dict[str, 'Clock'] = {}

from .loggers import get_logger


class Clock():
    def __new__(cls, name: str):
        if name in clocks:
            return clocks[name]
        new_clock = super().__new__(cls)
        new_clock.initialize(name)
        clocks[name] = new_clock
        return new_clock

    def initialize(self, name: str):
        self.logger = get_logger()
        self.name = name
        self.cumul = 0
        self.cumul_local = 0
        self.num_calls = 0
        self.num_calls_local = 0

    def __call__(self, func):
        if hasattr(func, 'clocked'):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                self.num_calls_local += 1
                self.cumul_local += time.time() - start
                return result
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                self.num_calls += 1
                self.cumul += time.time() - start
                return result

        wrapper.clocked = True

        return wrapper

    def report(self):
        num_calls = self.num_calls_local + self.num_calls
        if num_calls == 0:
            return
        tot_time = self.cumul + self.cumul_local
        avg_time = tot_time / num_calls * 1000

        self.logger.info(
            f'{self.name:>20s}   ({num_calls:>7d} CALLs): {tot_time:>13.4f} s  ({avg_time:>10.1f} ms/CALL)'
            )

    @staticmethod
    def report_all():
        total = Clock('total')
        for clock in clocks.values():
            clock.report()
            total.cumul += clock.cumul
            total.num_calls += clock.num_calls
