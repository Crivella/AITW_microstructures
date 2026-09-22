import json
import logging

from ..params import TrainParams
from .main import click, wood_microstructure

verbose_map = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}

@wood_microstructure.command()
@click.pass_context
@click.option('--output_dir', type=click.Path(), help='Output directory')
@click.option('-v', '--verbose', help='Verbose output', count=True)
@TrainParams.to_click_options
def train(ctx, config_file, output_dir, verbose) -> None:
    """Train wood microstructure model"""
    from wood_microstructure.surrogate_train import TrainSurrogate

    loglevel = verbose_map.get(verbose, logging.DEBUG)

    data = {}
    if config_file:
        with open(config_file, 'r') as f:
            data = json.load(f)

    overrides = ctx.obj.get('override_params', {})
    if overrides:
        data.update(overrides)

    TrainSurrogate.run_from_dict(data=data, output_dir=output_dir, loglevel=loglevel)

__all__ = [
    'train'
]
