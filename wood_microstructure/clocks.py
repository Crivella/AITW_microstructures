"""Implement clock to time the execution of functions."""

import time
from collections import defaultdict
from functools import wraps


class Clock():
    """Class to time the execution of functions.
    Usage:
    class MyClass(Clock):
        @Clock.register('my_function')
        def my_function(self):
            pass
    """
    CLOCK_MARKER = '__clock__'

    def __init__(self, *args, **kwargs):
        self.clocks = {'total': defaultdict(int)}
        super().__init__(*args, **kwargs)

    def __getattribute__(self, name):
        res = super().__getattribute__(name)
        if callable(res) and hasattr(res, Clock.CLOCK_MARKER) and not hasattr(res, '__wrapped__'):
            clock_lst = getattr(res, Clock.CLOCK_MARKER, None)
            if clock_lst is not None:
                clock_dcts = []
                for clock in clock_lst:
                    ptr = self.clocks.setdefault(clock, defaultdict(int))
                    clock_dcts.append(ptr)
                total_clock = self.clocks['total']
                @wraps(res)
                def wrapped(*args, **kwargs):
                    start = time.time()
                    result = res(*args, **kwargs)
                    delta = time.time() - start
                    for ptr in clock_dcts:
                        ptr['tot_time'] += delta
                        ptr['num_calls'] += 1

                    total_clock['tot_time'] += delta
                    total_clock['num_calls'] += 1
                    return result
                wrapped.__wrapped__ = res
                return wrapped
        return res

    def report_clocks(self) -> str:
        res = ['Time report:']
        all_tot_time = self.clocks['total']['tot_time']
        for name, clock in self.clocks.items():
            num_calls = clock['num_calls']

            tot_time = clock['tot_time']
            avg_time = (tot_time / num_calls * 1000) if num_calls > 0 else 0

            frc_string = ''
            if all_tot_time > 0:
                frc_time = (tot_time / all_tot_time * 100)
                frc_string = f'{frc_time:5.1f}% '

            msg = f'{name:>20s}   ({num_calls:>7d} CALLs):{frc_string}{tot_time:>13.4f} s  ({avg_time:>10.1f} ms/CALL)'
            res.append(msg)

        return '\n'.join(res)

    @staticmethod
    def register(name: str):
        def decorator(func):
            if not hasattr(func, Clock.CLOCK_MARKER):
                setattr(func, Clock.CLOCK_MARKER, [])
                func.__clock__ = []
            ptr = getattr(func, Clock.CLOCK_MARKER)
            ptr.insert(0, name)
            return func
        return decorator
