import os
import sys

import nrrd
import numpy as np

from .main import click, postproc


@postproc.command()
@click.argument('input_file', required=True, type=click.Path(exists=True))
@click.option(
    '--threshold',
    type=click.IntRange(0, 255),
    default=None,
    help='Threshold value for binarization (0-255)',
    )
def volume_npy_to_nrrd(input_file, threshold):
    """Convert a numpy volume file to nrrd format."""
    dirname = os.path.dirname(input_file)
    name, ext = os.path.splitext(os.path.basename(input_file))
    outfile = os.path.join(dirname, f'{name}.nrrd')

    data = np.ascontiguousarray(np.load(input_file))
    click.echo(f'Loaded volume data from `{input_file}` with shape {data.shape}')

    if threshold is not None:
        w = data > threshold
        data[w] = 1
        data[~w] = 0
        click.echo(f'Binarized volume data with threshold {threshold}')

    nrrd.write(
        outfile,
        data,
        index_order='C',
    )
    click.echo(f'Saved nrrd file to `{outfile}`')

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

__all__ = [
    'volume_npy_to_nrrd',
    'plot_volume',
]
