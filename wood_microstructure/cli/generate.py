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
@click.option(
    '--output-formats', type=str, required=False, default='tiff',
    help='Comma-separated list of output formats for the 2D slices (png, tiff), default: tiff'
)
@click.option('-v', '--verbose', help='Verbose output', count=True)
@click.option(
    '--num-parallel', type=int, default=1,
    help=(
        'Number of parallel processeses used for single microstructure generation. When used in conjunction with'
        ' --surrogate, defines the batch size for surrogate model inference.'
    )
)
@click.option(
    '--num-concurrent', type=int, default=1,
    help='Number of concurrent microstructure generations.'
)
@click.option('--surrogate/--no-surrogate', is_flag=True, default=False, help='Use surrogate model')
def generate(
        wood_type, json_file, output_dir,
        output_formats, verbose,
        num_concurrent, num_parallel,
        surrogate
    ) -> None:
    """Generate wood microstructure"""
    allowed_fmts = WoodMicrostructure.allowed_output_formats_2d
    output_formats = output_formats.replace(' ', '').split(',') if output_formats else ['tiff']
    output_formats = [fmt.lower() for fmt in filter(None, output_formats)]
    if not all(fmt in allowed_fmts for fmt in output_formats):
        raise ValueError(f"Invalid output format(s). Allowed formats: {allowed_fmts}")
    cls = wood_type_map.get(wood_type.lower())

    loglevel = verbose_map.get(verbose, logging.DEBUG)

    with open(json_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    args = [(d, output_dir, output_formats, loglevel, num_parallel) for d in data]

    if surrogate:
        for arg in args:
            arg[0]['surrogate'] = True

    if num_concurrent > 1:
        with mp.Pool(num_concurrent) as pool:
            pool.starmap(cls.run_from_dict, args)
    else:
        for arg in args:
            cls.run_from_dict(*arg)

    click.echo(f"Birch microstructures generated and saved to `{output_dir or 'current directory'}`")

__all__ = [
    'generate'
]
