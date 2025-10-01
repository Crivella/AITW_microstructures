import json
import logging
import multiprocessing as mp

from .. import BirchMicrostructure, SpruceMicrostructure
from ..microstructure import WoodMicrostructure
from .main import click, wood_microstructure

verbose_map = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}

wood_type_map: dict[str, WoodMicrostructure] = {
    'spruce': SpruceMicrostructure,
    'birch': BirchMicrostructure,
}

@wood_microstructure.command()
@click.argument('wood_type', required=True, type=click.Choice(['spruce', 'birch'], case_sensitive=False))
@click.argument('json_file', required=True, type=click.Path(exists=True))
@click.option('--output_dir', type=click.Path(), help='Output directory')
@click.option('-v', '--verbose', help='Verbose output', count=True)
# @click.option('--log_file', type=click.Path(), help='Log file name')
@click.option('--max-parallel', type=int, default=1, help='Max parallel processeses')
def generate(wood_type, json_file, output_dir, verbose, max_parallel):
    """Generate wood microstructure"""
    cls = wood_type_map.get(wood_type.lower())

    loglevel = verbose_map.get(verbose, logging.DEBUG)

    with open(json_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    args = [(d, output_dir, loglevel) for d in data]

    with mp.Pool(max_parallel) as pool:
        pool.starmap(cls.run_from_dict, args)

    click.echo(f"Birch microstructure generated and saved to `{output_dir or 'current directory'}`")

__all__ = [
    'generate'
]
