import json
import logging
import sys

import nrrd
import numpy as np

from ..myio import read_volume, write_volume
from ..params import FitPorosityParams
from .main import click, postproc

verbose_map = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}

@postproc.command()
@click.argument('input_file', required=True, type=click.Path(exists=True))
@click.argument('output_file', required=True, type=click.Path())
def volume_convert_format(input_file, output_file):
    """Convert a volume file between formats (npy, nrrd, vti)."""
    data = read_volume(input_file)
    click.echo(f'Loaded volume data from `{input_file}` with shape {data.shape}')

    write_volume(output_file, data)
    click.echo(f'Saved volume data to `{output_file}`')

@postproc.command()
@click.argument('input_file', required=True, type=click.Path(exists=True))
@click.option(
    '--threshold',
    type=click.IntRange(0, 255),
    default=125,
    help='Threshold value for binarization (0-255)',
)
def plot_volume(input_file, threshold):
    """Plot a numpy volume file using mayavi."""
    try:
        import matplotlib.pyplot as plt
        from mayavi import mlab
        from tvtk.util import ctf
    except ImportError:
        click.echo('Please install the package with the extra [utils] dependency to use this feature.')
        sys.exit(1)
    if input_file.endswith('.nrrd'):
        data, header = nrrd.read(input_file, index_order='C')
    elif input_file.endswith('.npy'):
        data = np.load(input_file)
    else:
        click.echo('Unsupported file format. Please provide a .nrrd or .npy file.')
        sys.exit(1)
    click.echo(f'Loaded volume data from `{input_file}` with shape {data.shape}')

    w = data > threshold
    data[w] = 1
    data[~w] = 0
    click.echo(f'Binarized volume data with threshold {threshold}')

    mlab.figure(bgcolor=(1.0, 1.0, 1.0), size=(1600, 1600))
    src = mlab.pipeline.scalar_field(data)
    src.update_image_data = True
    volume = mlab.pipeline.volume(src, vmin=0, vmax=1)

    c = ctf.save_ctfs(volume._volume_property)
    c['rgb'] = plt.get_cmap('gray')(np.arange(2))
    ctf.load_ctfs(c, volume._volume_property)

    volume.update_ctf = True

    mlab.axes()
    mlab.show()

@postproc.command()
@click.pass_context
@click.option('--output_dir', type=click.Path(), help='Output directory')
@click.option('-v', '--verbose', help='Verbose output', count=True)
@FitPorosityParams.to_click_options
def filter_porosity(ctx, config_file, output_dir, verbose) -> None:
    """Geometry filter for OpenLB interpolated boundary conditions.

    \b
    Features:
    - SDF-based porosity control: --porosity with --sdf-sigma for "acid treatment" simulation
    - Grayscale-aware processing: treats raw intensity as continuous density field
    - Downsampling and smoothing: efficient multi-resolution processing
    - Supersampling: high-quality upsampling via anti-aliasing
    - Padding/cropping: add pore space or crop geometry
    - Validation: checks for isolated voxels and geometry quality
    Binary convention: 0 = pore, 255 = solid
    """
    from wood_microstructure.filter_fit_porosity import FitPorosity

    loglevel = verbose_map.get(verbose, logging.DEBUG)

    data = {}
    if config_file:
        with open(config_file, 'r') as f:
            data = json.load(f)

    overrides = ctx.obj.get('override_params', {})
    if overrides:
        data.update(overrides)

    FitPorosity.run_from_dict(data=data, output_dir=output_dir, loglevel=loglevel)


__all__ = [
    'volume_convert_format',
    'plot_volume',
    'filter_porosity',
]
