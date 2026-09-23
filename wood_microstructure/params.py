"""Input paramemters"""
import json
from collections.abc import Callable
from copy import copy
from dataclasses import Field, dataclass, field, fields
from typing import ClassVar, Self

import numpy as np
import numpy.typing as npt
import rich_click as click
from click import ParamType
from click.core import ParameterSource


class DelimitedList(click.ParamType):
    """A custom Click parameter type that parses a comma-separated list of values."""

    name = 'list'

    def __init__(self, delimiter=',', subtype=str, exact_length=None):
        super().__init__()
        self.delimiter = delimiter
        self.subtype = subtype
        self.exact_length = exact_length

    def convert(self, value, param, ctx):
        if isinstance(value, list):
            res = [self.subtype(item) for item in value]
        else:
            try:
                items = value.split(self.delimiter)
                res = [self.subtype(item.strip()) for item in items]
            except Exception as e:
                self.fail(f"Could not parse list: {e}", param, ctx)

        if self.exact_length is not None and len(res) != self.exact_length:
            self.fail(f"Expected exactly {self.exact_length} items, got {len(res)}", param, ctx)


@dataclass
class JsonParams:
    params_map: ClassVar[dict[str, str]] = {}
    post_set: ClassVar[list[str]] = []

    def _to_json(self, data: dict) -> dict:
        """Convert the parameters to a JSON serializable dictionary"""
        return data

    def to_json(self, json_file: str):
        """Save the parameters to a JSON file"""
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        data = self._to_json(data)

        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, json_file: str) -> list[Self]:
        """Create an instance from a JSON file"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        res = []
        if isinstance(data, dict):
            res = [cls.from_dict(data)]
        elif isinstance(data, list):
            res = [cls.from_dict(item) for item in data]
        else:
            raise ValueError('Invalid data format in JSON file')
        return res

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create an instance from a JSON file"""
        post = {}
        data = {cls.params_map.get(k, k): v for k, v in data.items()}
        for k in cls.post_set:
            if k in data:
                post[k] = data.pop(k)
        res = cls(**data)
        for k, v in post.items():
            setattr(res, k, v)
        return res

    @classmethod
    def to_click_options(cls, func: Callable) -> Callable:
        """Decorator to add click options for the parameters to a click command"""
        name_map = {}

        OVERRIDE_GROUP = 'Override Input Parameters'

        groups = {'Options'}

        def callback(ctx: click.Context, param: click.Parameter, value):
            ctx.ensure_object(dict)
            source = ctx.get_parameter_source(param.name)
            if source == ParameterSource.DEFAULT:
                return
            name = name_map.get(param.name, param.name)
            overrides = ctx.obj.setdefault('override_params', {})
            overrides[name] = value

        for fld in fields(cls)[::-1]:
            extra_help = ''
            metadata = getattr(fld, 'metadata', {})
            if not fld.init:
                continue
            if fld.name.startswith('_'):
                continue
            if fld.type not in (
                    int, float, str, bool,
                    list[str], tuple[int, int, int]
                ):
                continue

            typ = fld.type
            if typ in (int, float):
                min_val = metadata.get('min', None)
                max_val = metadata.get('max', None)
                if min_val is not None or max_val is not None:
                    cls_typ = click.FloatRange if typ == float else click.IntRange
                    typ = cls_typ(min=min_val, max=max_val)
            elif typ == str:
                if metadata.get('file', False):
                    typ = click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True)
                elif metadata.get('dir', False):
                    typ = click.Path(exists=True, file_okay=False, readable=True, resolve_path=True)
            elif typ == list[str]:
                typ = DelimitedList()
                extra_help = ' (comma-separated list)'
            elif typ == tuple[int, int, int]:
                typ = DelimitedList(subtype=int, exact_length=3)
                extra_help = ' (comma-separated list of 3 integers)'

            expose = metadata.get('expose_value', False)
            # prefix = '--param-' if not expose else '--'
            prefix = '--'

            decl = f'{prefix}{fld.name}'
            if fld.type == bool:
                decl = f'{prefix}{fld.name}/{prefix}no-{fld.name}'
            name_map[f'param_{fld.name}'] = fld.name

            required = metadata.get('required', False)

            group = metadata.get('group', OVERRIDE_GROUP)
            groups.add(group)

            help_str = metadata.get('help', None)
            if help_str is not None and extra_help:
                help_str += extra_help

            kwargs = {
                'type': typ,
                'is_flag': fld.type == bool,
                'required': required,
                'expose_value': expose,
                'callback': callback,
                'help': help_str,
                'panel': group,
            }

            if 'flag_value' in metadata:
                kwargs['flag_value'] = metadata['flag_value']

            func = click.option(decl, **kwargs)(func)

        func = click.option(
            '--config-file', type=click.Path(exists=True), help='Path to file with parameters'
        )(func)

        for group in sorted(groups, key=lambda x: (x != 'Options', x))[::-1]:
            func = click.option_panel(group)(func)

        return func

@dataclass
class BaseWoodParams(JsonParams):
    """Base class for parameters"""
    # period_parameter: int  # This parameter is related to the period of the year ring size
    period_parameter: int = field(metadata={'help': 'This parameter is related to the period of the year ring size'})

    cell_r: float = field(metadata={'help': 'The grid distance for the nodes we generated. In Unit of voxels.'})
    cell_length: float = field(metadata={'help': 'Average fiber length'})
    cell_length_variance: float = field(metadata={'help': 'Standard deviation of fiber length'})
    cell_wall_thick: float = field(metadata={'help': 'Cell wall thickness'})
    cell_end_thick: int = field(metadata={'help': 'End of cell wall thickness along L direction'})

    ray_height: float = field(metadata={'help': 'The width of the ray cell'})
    ray_space: float = field(
        metadata={'help': 'The space between ray cell along T direction. The distance is raySpace*cellR'}
    )
    ray_cell_length: float = field(metadata={'help': 'Ray cell length along radial direction'})
    ray_cell_variance: float = field(metadata={'help': 'Ray cell length deviation along radial direction'})
    ray_cell_num: float = field(metadata={'help': 'Ray cell count in a group'})
    ray_cell_num_std: float = field(metadata={'help': 'Ray cell count in a group'})

    vessel_length: float = field(metadata={'help': 'Average vessel length'})
    vessel_length_variance: float = field(metadata={'help': 'Standard deviation of fiber length'})
    vessel_thicker: int = field(metadata={'help': 'Assume vessel is thicker than ray cells'})
    vessel_count: int = field(metadata={'help': 'This number is used to control the vessel number and distribution.'})

    is_exist_vessel: bool = field(default=True, metadata={'help': 'Whether to generate vessel cells'})
    is_exist_ray_cell: bool = field(default=True, metadata={'help': 'Whether to generate ray cells'})

    random_seed: int = field(default=42, metadata={'help': 'Random seed initialization for reproducibility'})

    size_volume: tuple[int, int, int] = field(
        default=(500, 500, 200),
        metadata={'help': 'The size of the volume to be generated'}
    )
    extra_size: tuple[int, int, int] = field(
        default=(150, 200, 100),
        metadata={'help': 'Extra size for the enlarged image'}
    )
    slice_interest_space: int = field(
        default=100,
        metadata={'help': 'We generate one slice every XXX slices to add random noise before interpolation.'}
    )

    apply_local_deform: bool = field(default=True, metadata={'help': 'Whether to apply local deformation'})
    apply_global_deform: bool = field(default=True, metadata={'help': 'Whether to apply global deformation'})

    save_slices_as_2d: bool = field(default=True, metadata={'help': 'Whether to save slices as 2D images'})
    save_volume_as_3d: bool = field(default=True, metadata={'help': 'Whether to save volume as 3D image'})
    save_volume_format: str = field(default='nrrd', metadata={'help': 'Format to save volume data (e.g., nrrd, npy)'})
    save_local_dist: bool = field(default=True, metadata={'help': 'Whether to save local deformation data'})
    save_global_dist: bool = field(default=True, metadata={'help': 'Whether to save global deformation data'})

    surrogate: bool = field(
        default=False,
        metadata={
            'help': 'Whether to use surrogate model for local deformation',
        },
    )
    weight_file: str = field(
        default=None,
        metadata={
            'help': 'Path to the weight file for the surrogate model',
            'file': True,
        },
    )
    binarize_threshold: int = field(
        default=None, metadata={
            'help': 'Threshold for binarization of the final volume data. If None, no bin is applied.',
            'min': 0, 'max': 255,
            'group': 'Post-processing Options',
        }
    )

    fit_porosity: bool = field(
        default=False,
        metadata={
            'help': 'Perform post-processing to fit porosity on the final volume data',
            'group': 'Post-processing Options',
        }
    )
    fit_porosity_config: str = field(
        default=None,
        metadata={
            'help': 'Path to file with parameters for fitting porosity',
            'group': 'Post-processing Options',
            'file': True,
        }
    )
    # Not user defined
    neighbor_local = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]], dtype=int)  # d-indices of the neighbor grid nodes

    # Internal parameters
    _all_slices = False  # Whether to save all slices or not
    _size_im_enlarge: tuple[int, int, int] = None
    _x_vector: npt.NDArray = None
    _y_vector: npt.NDArray = None
    _grid: tuple[npt.NDArray, npt.NDArray] = None
    _num_grid_nodes: int = None

    _save_slice: list[int] | str = None  # List of slices (Z-index) to save (NOTE: inputfile is 1-indexed)
    _save_slice_map: dict[int, int] = None

    params_map = {
        'sizeVolume': 'size_volume',
        'saveSlice': 'save_slice',
        'cellR': 'cell_r',
        'cellLength': 'cell_length',
        'cellLengthVariance': 'cell_length_variance',
        'rayCellLength': 'ray_cell_length',
        'rayCell_variance': 'ray_cell_variance',
        'rayCellNum': 'ray_cell_num',
        'rayCellNumStd': 'ray_cell_num_std',
        'vesselLength': 'vessel_length',
        'vesselLengthVariance': 'vessel_length_variance',
        'raywHeight': 'ray_height',
        'raySpace': 'ray_space',
        'isExistVessel': 'is_exist_vessel',
        'isExistRayCell': 'is_exist_ray_cell',
        'cellWallThick': 'cell_wall_thick',
        'writeGlobalDeformData': 'save_global_dist',
        'writeLocalDeformData': 'save_local_dist',
    }

    post_set = ['save_slice']

    @property
    def save_slice(self):
        """List of slices (Z-index) to save (NOTE: inputfile is 1-indexed)"""
        if self._save_slice is None:
            self.save_slice = 'all'  # Default to save all slices
        return self._save_slice

    @save_slice.setter
    def save_slice(self, value: list[int] | str):
        self._all_slices = False
        if isinstance(value, str):
            if value == 'all':
                self._all_slices = True
                value = tuple(range(self.size_im_enlarge[2]))
            else:
                value = (int(value) - 1,)
        elif isinstance(value, int):
            value = (value - 1,)
        elif isinstance(value, (tuple, list)):
            value = tuple(int(s) - 1 for s in value)

        self._save_slice = value

    @property
    def all_slices(self):
        """Whether to save all slices"""
        return self._all_slices

    @property
    def save_slice_map(self):
        """Map of saved slices"""
        if self._save_slice_map is None:
            self._save_slice_map = {s: i for i, s in enumerate(self.save_slice)}
        return self._save_slice_map

    @property
    def size_im_enlarge(self):
        """Size of enlarged image"""
        if self._size_im_enlarge is None:
            self._size_im_enlarge = np.array(self.size_volume) + np.array(self.extra_size)
        return self._size_im_enlarge

    @property
    def x_vector(self):
        """X vector"""
        if self._x_vector is None:
            self._x_vector = np.arange(5, self.size_im_enlarge[0] - 4, self.cell_r)  # Right-inclusive
        return self._x_vector

    @property
    def y_vector(self):
        """Y vector"""
        if self._y_vector is None:
            self._y_vector = np.arange(5, self.size_im_enlarge[1] - 4, self.cell_r)  # Right-inclusive
        return self._y_vector

    @property
    def grid(self):
        """Tuple of X and Y  2D grid"""
        if self._grid is None:
            self._grid = np.meshgrid(self.x_vector, self.y_vector, indexing='ij')
        return self._grid

    @property
    def x_grid(self):
        """X coordinate of the 2D grid (slice of 3D)"""
        return self.grid[0]

    @property
    def y_grid(self):
        """Y coordinate of the 2D grid (slice of 3D)"""
        return self.grid[1]

    @property
    def num_grid_nodes(self):
        """Number of grid nodes"""
        if self._num_grid_nodes is None:
            self._num_grid_nodes = self.x_grid.size()
        return self._num_grid_nodes

    @property
    def fit_porosity_params(self):
        if self.fit_porosity_config is None:
            return {}
        with open(self.fit_porosity_config, 'r') as f:
            data = json.load(f)
        return data

    def _to_json(self, data: dict) -> dict:
        """Convert the parameters to a JSON serializable dictionary"""
        if self.all_slices:
            data['saveSlice'] = 'all'
        else:
            data['saveSlice'] = [s + 1 for s in self.save_slice]
        return data

def with_default(field_name: str, default, **kwargs) -> Field:
    """Return a new field of `BaseParams` with the default value replaced"""
    fld: Field = BaseWoodParams.__dataclass_fields__[field_name]
    new = copy(fld)
    new.default = default
    for key, value in kwargs.items():
        setattr(new, key, value)
    return new

@dataclass
class BirchParams(BaseWoodParams):
    """Define the parameters for birch"""
    period_parameter: int = with_default('period_parameter', default=0)

    cell_r: float = with_default('cell_r', default=14.5)
    cell_length: float = with_default('cell_length', default=2341)
    cell_length_variance: float = with_default('cell_length_variance', default=581)
    cell_wall_thick: float = with_default('cell_wall_thick', default=2)
    cell_end_thick: int = with_default('cell_end_thick', default=4)

    ray_height: float = with_default('ray_height', default=42)
    ray_space: float = with_default('ray_space', default=20)
    ray_cell_length: float = with_default('ray_cell_length', default=62)
    ray_cell_variance: float = with_default('ray_cell_variance', default=15)
    ray_cell_num: float = with_default('ray_cell_num', default=11.33)
    ray_cell_num_std: float = with_default('ray_cell_num_std', default=3.39)

    vessel_length: float = with_default('vessel_length', default=780)
    vessel_length_variance: float = with_default('vessel_length_variance', default=195)
    vessel_thicker: int = with_default('vessel_thicker', default=1)
    vessel_count: int = with_default('vessel_count', default=50)

    is_exist_vessel: bool = with_default('is_exist_vessel', default=True)
    is_exist_ray_cell: bool = with_default('is_exist_ray_cell', default=True)


@dataclass
class SpruceParams(BaseWoodParams):
    """Define the parameters for spruce"""
    period_parameter: int = with_default('period_parameter', default=1000)

    cell_r: float = with_default('cell_r', default=14.5)
    cell_length: float = with_default('cell_length', default=4877)
    cell_length_variance: float = with_default('cell_length_variance', default=1219)
    cell_wall_thick: float = with_default('cell_wall_thick', default=3)
    cell_end_thick: int = with_default('cell_end_thick', default=2)

    ray_height: float = with_default('ray_height', default=40)
    ray_space: float = with_default('ray_space', default=0)
    ray_cell_length: float = with_default('ray_cell_length', default=149.4)
    ray_cell_variance: float = with_default('ray_cell_variance', default=38.5)
    ray_cell_num: float = with_default('ray_cell_num', default=8.44)
    ray_cell_num_std: float = with_default('ray_cell_num_std', default=4.39)

    vessel_length: float = with_default('vessel_length', default=4877)
    vessel_length_variance: float = with_default('vessel_length_variance', default=1219)
    vessel_thicker: int = with_default('vessel_thicker', default=0)
    vessel_count: int = with_default('vessel_count', default=0)

    is_exist_vessel: bool = with_default('is_exist_vessel', default=True)
    is_exist_ray_cell: bool = with_default('is_exist_ray_cell', default=True)

@dataclass
class FitPorosityParams(JsonParams):
    """Define the parameters for fitting porosity"""
    input_file: str = field(
        default=None,
        metadata={
            'help': 'Input file path to a volume data file.',
            'group': 'Options',
            # 'expose_value': True
            'file': True,
        }
    )

    threshold: float = field(
        default=127,
        metadata={
            'help': 'Threshold for binarization of the final volume data. If None, no bin is applied.',
            'min': 0, 'max': 255,
        }
    )
    solid_is_high: bool = field(
        default=True,
        metadata={'help': 'Whether the solid part is dark or light in the image. True=solid is light.'}
    )
    down: int = field(
        default=2,
        metadata={
            'help': 'Downsample factor (1=no downsampling)',
            'min': 1,
        }
    )

    smooth_low_iters: int = field(
        default=0, metadata={
            'help': 'Number of iterations for low-pass smoothing. If <= 0, no smoothing is applied.',
            'min': 0,
        }
    )
    smooth_sigma: float = field(
        default=0.6, metadata={
            'help': 'Sigma for Gaussian smoothing. If <= 0, no smoothing is applied.',
            'min': 0,
        }
    )

    thicken: int = field(default=0, metadata={'help': 'Dilate solid by N voxels (thicken walls, narrow throats)'})
    final_smooth_iters: int = field(
        default=0,
        metadata={
            'help': 'Number of iterations for final low-pass smoothing. If <= 0, no smoothing is applied.',
            'min': 0,
        }
    )
    final_smooth_sigma: float = field(
        default=0.5,
        metadata={
            'help': 'Sigma for final Gaussian smoothing. If <= 0, no smoothing is applied.',
            'min': 0,
        }
    )

    upsample_intermediate: float = field(
        default=None, metadata={'help': 'Supersampling: first upsample by this factor (e.g., 3)'}
    )
    upsample_final: float = field(
        default=None,
        metadata={
            'help': (
                'Supersampling: then downsample to this net factor (e.g., 2). '
                'Creates smoother upsampling via anti-aliasing.'
            )
        }
    )

    pad: int = field(
        default=0,
        metadata={
            'help': (
                'Add N voxels of pore padding (positive) or crop N voxels (negative). '
                'Example: `pad=10` adds padding, `pad=-5` crops 5 voxels from edges.'
            )
        }
    )
    pad_xy_only: bool = field(default=False, metadata={'help': 'Only apply padding in XY directions, not Z.'})

    porosity: float = field(
        default=None,
        metadata={
            'help': (
                'Target porosity [0, 1]. Uses SDF threshold search on raw grayscale. '
                'Pass \'None\' or omit to disable (use `threshold` instead).'
            ),
            'min': 0, 'max': 1,
        }
    )
    sdf_sigma: float = field(
        default=0.5,
        metadata={
            'help': 'SDF Gaussian blur sigma (higher = more boundary smearing)',
            'min': 0,
        },
    )
    adjust_porosity_post: bool = field(
        default=False, metadata={'help': 'Fine-tune porosity after processing via post-processing SDF adjustment'}
    )

    majority_filter: int = field(
        default=1,
        metadata={
            'help': 'Remove thin protrusions (iterations). 0=off, 2-3 for aggressive filtering.',
            'min': 0,
        }
    )

@dataclass
class TrainParams(JsonParams):
    """Define the parameters for training the surrogate model"""
    train_dir: str = field(
        default=None,
        metadata={
            'help': 'Directory containing training data',
            'group': 'Dataset Options',
            'dir': True,
            'required': True,
        }
    )
    validation_dir: str = field(
        default=None,
        metadata={
            'help': (
                'Directory containing validation data. '
                'IF NOT PROVIDED, training data will be split for validation.'
            ),
            'group': 'Dataset Options',
            'dir': True,
            # 'required': True,
        }
    )
    test_dir: str = field(
        default=None,
        metadata={
            'help': 'Directory containing test data. IF NOT PROVIDED, training data will be split for testing.',
            'group': 'Dataset Options',
            'dir': True,
            # 'required': True,
        }
    )

    backbone_subdir: str = field(
        default='volImgBackBone',
        metadata={
            'help': 'Subdirectory name for the undistorted backbone images within the train/validation/test directories',
            'group': 'Dataset Options',
        }
    )
    distorted_subdir: str = field(
        default='LocalDistVolume',
        metadata={
            'help': 'Subdirectory name for the distorted images within the train/validation/test directories',
            'group': 'Dataset Options',
        }
    )
    u_map_subdir: str = field(
        default='LocalDistVolumeDispU',
        metadata={
            'help': 'Subdirectory name for the u displacement maps within the train/validation/test directories',
            'group': 'Dataset Options',
        }
    )
    v_map_subdir: str = field(
        default='LocalDistVolumeDispV',
        metadata={
            'help': 'Subdirectory name for the v displacement maps within the train/validation/test directories',
            'group': 'Dataset Options',
        }
    )

    learning_rate: float = field(
        default=2e-4,
        metadata={
            'help': 'Learning rate for training the surrogate model',
            'group': 'Training Options',
            'min': 1e-8, 'max': 1.0,
        }
    )
    epochs: int = field(
        default=1000,
        metadata={
            'help': 'Number of epochs for training the surrogate model',
            'group': 'Training Options',
            'min': 1,
        }
    )
    training_batch_size: int = field(
        default=2,
        metadata={
            'help': 'Batch size for training the surrogate model',
            'group': 'Training Options',
            'min': 1,
        }
    )
    validation_batch_size: int = field(
        default=2,
        metadata={
            'help': 'Batch size for validation during training the surrogate model',
            'group': 'Training Options',
            'min': 1,
        }
    )
    patience: int = field(
        default=20,
        metadata={
            'help': 'Number of epochs with no improvement after which training will be stopped (early stopping)',
            'group': 'Training Options',
            'min': 1,
        }
    )
    training_workers: int = field(
        default=4,
        metadata={
            'help': 'Number of worker threads for loading training data',
            'group': 'Training Options',
            'min': 1,
        }
    )
    validation_workers: int = field(
            default=4,
            metadata={
                'help': 'Number of worker threads for loading validation data',
                'group': 'Training Options',
                'min': 1,
            }
        )
    save_interval: int = field(
        default=100,
        metadata={
            'help': 'Save model checkpoint every N epochs. Set to 0 to disable checkpoint saving.',
            'group': 'Training Options',
            'min': 0,
        }
    )

    pretrain_weights: str = field(
        default=None,
        metadata={
            'help': 'Path to pre-trained model weights to initialize training. If None, training starts from scratch.',
            'group': 'Transfer Options',
            'file': True,
        }
    )
    frozen_layers: list[str] = field(
        default_factory=list,
        metadata={
            'help': 'List of layer names to freeze during training. If empty, all layers are trainable.',
            'group': 'Transfer Options',
        }
    )

    cross_validation: bool = field(
        default=False,
        metadata={
            'help': 'Whether to use cross-validation for training the surrogate model',
            'group': 'Cross-Validation Options',
        }
    )
    num_folds: int = field(
        default=5,
        metadata={
            'help': 'Number of folds for cross-validation',
            'group': 'Cross-Validation Options',
            'min': 1,
        }
    )
    train_ratio: float = field(
        default=0.9,
        metadata={
            'help': 'Ratio of training data in each fold for cross-validation',
            'group': 'Cross-Validation Options',
            'min': 0.0, 'max': 1.0,
        }
    )
