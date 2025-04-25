"""Input paramemters"""
import json
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class BaseParams:
    """Base class for parameters"""
    period_parameter: int  # This parameter is related to the period of the year ring size

    cell_r: float  # The grid distance for the nodes we generated. In Unit of voxels.
    cell_length: float  # Average fiber length
    cell_length_variance: float  # Standard deviation of fiber length
    cell_wall_thick: float  # Cell wall thickness
    cell_end_thick: int  # End of cell wall thickness along L direction

    ray_height: float  # The width of the ray cell
    ray_space: float  # The space between ray cell along T direction. The distance is raySpace*cellR
    ray_cell_length: float  # Ray cell length along radial direction
    ray_cell_variance: float  # Ray cell length deviation along radial direction
    ray_cell_num: float  # Ray cell count in a group
    ray_cell_num_std: float  # Ray cell count in a group

    vessel_length: float  # Average vessel length
    vessel_length_variance: float  # Standard deviation of fiber length
    vessel_thicker: int  # Assume vessel is thicker than ray cells
    vessel_count: int  # This number is used to control the vessel number and distribution.

    is_exist_vessel: bool = True  # Whether to generate vessel cells
    is_exist_ray_cell: bool = True  # Whether to generate ray cells

    random_seed: int = 42  # Random seed initialization for reproducibility

    size_volume: tuple[int, int, int] = (500, 500, 200)  # The size of the volume to be generated
    extra_size: tuple[int, int, int] = (150, 200, 100)  # Extra size for the enlarged image
    slice_interest_space: int = 100  # We generate one slice every XXX slices to add random noise before interpolation.

    save_slice: list[int] = 1  # List of slices (Z-index) to save (NOTE: inputfile is 1-indexed)
    # save_volume_as_3d: bool = True
    # write_local_deform_data: bool = True
    # write_global_deform_data: bool = False

    # Not user defined
    neighbor_local = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]], dtype=int)  # d-indices of the neighbor grid nodes

    # Internal parameters
    _size_im_enlarge: tuple[int, int, int] = None
    _x_vector: npt.NDArray = None
    _y_vector: npt.NDArray = None
    _grid: tuple[npt.NDArray, npt.NDArray] = None
    _num_grid_nodes: int = None

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
    }

    def __post_init__(self):
        """Post-initialization"""
        ss = self.save_slice
        if isinstance(ss, str):
            ss = (int(ss) - 1,)
        elif isinstance(ss, int):
            ss = (ss - 1,)
        elif isinstance(ss, (tuple, list)):
            ss = tuple(int(s) - 1 for s in ss)
        self.save_slice = ss

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

    def to_json(self, json_file: str):
        """Save the parameters to a JSON file"""
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, json_file: str) -> list['BirchParams']:
        """Create an instance from a JSON file"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        res = []
        if isinstance(data, dict):
            data = {cls.params_map.get(k, k): v for k, v in data.items()}
            res = [cls(**data)]
        elif isinstance(data, list):
            for item in data:
                item = {cls.params_map.get(k, k): v for k, v in item.items()}
                res.append(cls(**item))
        else:
            raise ValueError('Invalid data format in JSON file')
        return res

    @classmethod
    def from_dict(cls, data: dict) -> 'BirchParams':
        """Create an instance from a JSON file"""
        data = {cls.params_map.get(k, k): v for k, v in data.items()}
        return cls(**data)

@dataclass
class BirchParams(BaseParams):
    """Define the parameters for birch"""
    period_parameter: int = 0

    cell_r: float = 14.5
    cell_length: float = 2341
    cell_length_variance: float = 581
    cell_wall_thick: float = 2
    cell_end_thick: int = 4

    ray_height: float = 42
    ray_space: float = 20
    ray_cell_length: float = 62
    ray_cell_variance: float = 15
    ray_cell_num: float = 11.33
    ray_cell_num_std: float = 3.39

    vessel_length: float = 780
    vessel_length_variance: float = 195
    vessel_thicker: int = 1
    vessel_count: int = 50

    is_exist_vessel: bool = True
    is_exist_ray_cell: bool = True


@dataclass
class SpruceParams(BaseParams):
    """Define the parameters for spruce"""
    period_parameter: int = 1000

    cell_r: float = 14.5
    cell_length: float = 4877
    cell_length_variance: float = 1219
    cell_wall_thick: float = 3
    cell_end_thick: int = 2

    ray_height: float = 40
    ray_space: float = 0
    ray_cell_length: float = 149.4
    ray_cell_variance: float = 38.5
    ray_cell_num: float = 8.44
    ray_cell_num_std: float = 4.39

    vessel_length: float = 4877
    vessel_length_variance: float = 1219
    vessel_thicker: int = 0
    vessel_count: int = 0

    is_exist_vessel: bool = True
    is_exist_ray_cell: bool = True
