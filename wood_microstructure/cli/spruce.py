import json
import logging
import multiprocessing as mp

import click

from .. import SpruceMicrostructure, SpruceParams
from ..loggers import set_console_level
from .main import wood_microstructure

verbose_map = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}

def run_from_dict(data: dict, output_dir: str = None, loglevel: int = logging.DEBUG):
    """Run the birch microstructure generation from a dictionary."""
    params = SpruceParams.from_dict(data)
    ms = SpruceMicrostructure(params, outdir=output_dir)
    ms.set_console_level(loglevel)
    ms.generate()



@wood_microstructure.command()
@click.option('--json_file', type=click.Path(exists=True), help='JSON file with parameters')
@click.option('--output_dir', type=click.Path(), help='Output directory')
@click.option('-v', '--verbose', help='Verbose output', count=True)
# @click.option('--log_file', type=click.Path(), help='Log file name')
@click.option('--max-parallel', type=int, default=1, help='Max parallel processeses')
def spruce(json_file, output_dir, verbose, max_parallel):
    """Generate spruce microstructure"""
    if json_file is None:
        raise ValueError('JSON file is required')

    loglevel = verbose_map.get(verbose, logging.DEBUG)
    # set_console_level(logger, verbose_map.get(verbose, logging.DEBUG))
    # if log_file:
    #     add_file_logger(logger, log_file)

    with open(json_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    args = [(d, output_dir, loglevel) for d in data]

    with mp.Pool(max_parallel) as pool:
        pool.starmap(run_from_dict, args)

    click.echo(f"Birch microstructure generated and saved to `{output_dir or 'current directory'}`")
