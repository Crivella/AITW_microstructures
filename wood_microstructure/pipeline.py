"""
Generic pipeline for running blocks of codes
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .clocks import Clock
from .loggers import LoggerMixin
from .params import JsonParams
from .progress import RichMixin

O = TypeVar('O', bound=JsonParams)


class Pipeline(RichMixin, LoggerMixin, Clock, ABC, Generic[O]):
    ParamsClass: O = None
    parsed_params_name: str = 'params.json'

    def __init__(self, params: O, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tasks = []

        self.init_params(params)
        self.init_pipeline()

    def init_params(self, params: O):
        """Initialize parameters"""
        self.params = params

        save_param_file = os.path.join(self.root_dir, self.parsed_params_name)
        self.params.to_json(save_param_file)

    @abstractmethod
    def init_pipeline(self):
        """Initialize pipeline"""

    def run_pipeline(self):
        """Run the pipeline of tasks"""
        cls_name = self.__class__.__name__
        op = self.overall_progress
        sp = self.step_progress

        idx = 0
        success_colors = ['green', 'bold green']
        self.overall_task_id = ot_id = op.add_task(f'Generating {cls_name} ...', total=len(self.tasks))
        with self.rich_live:
            for func, args, kwargs, logtask in self.tasks:
                self.logger.debug('=' * 80)
                self.logger.debug('Running task: %s', func.__name__)
                self.logger.debug('Task args: %s', args)
                self.logger.debug('Task kwargs: %s', kwargs)
                if logtask:
                    step_id = sp.add_task(f'{func.__name__:>30s}', total=1)

                func(*args, **kwargs)

                if logtask:
                    color = success_colors[idx % len(success_colors)]
                    sp.advance(step_id, 1)
                    sp.update(step_id, description=f'[{color}]{func.__name__:>30s}')
                    idx += 1

                op.update(ot_id, advance=1)

    def report(self):
        """Final report for the generation"""
        self.logger.info(self.report_clocks())

    def run(self):
        """Generate the volume image"""
        self.run_pipeline()
        self.report()

    @classmethod
    def run_from_dict(
            cls, *args,
            data: dict, loglevel: int = logging.DEBUG,
            **kwargs
        ) -> None:
        """Run the generator from a dictionary of parameters"""
        if args:
            raise ValueError('Positional arguments are not supported for run_from_dict. Use keyword arguments instead.')
        params = cls.ParamsClass.from_dict(data)
        ffp = cls(params, *args, **kwargs)
        ffp.set_console_level(loglevel)
        ffp.run()
