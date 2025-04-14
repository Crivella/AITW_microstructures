"""Input paramemters"""
import json
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class RayCellParams:
    """Ray cell parameters"""
    random_seed: int = 42

    size_volume: tuple[int, int, int] = (500, 1200, 300)

    cell_r: float = 14.5
    cell_length: float = 2341
    cell_length_variance: float = 581
    cell_wall_thick: float = 2

    ray_height: float = 42
    ray_space: float = 20

    ray_cell_length: float = 62
    ray_cell_variance: float = 15
    ray_cell_num: float = 11.33
    ray_cell_num_std: float = 3.39

    vessel_length: float = 780
    vessel_length_variance: float = 195

    is_exist_vessel: bool = True
    is_exist_ray_cell: bool = True

    save_slice: int = 1
    save_volume_as_3d: bool = True
    write_local_deform_data: bool = True
    write_global_deform_data: bool = False

    # Not user defined
    slice_interest_space: int = 100
    cell_end_thick: int = 4
    neighbor_local = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]], dtype=int)
    vessel_count: int = 50

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
    def size_im(self):
        """Size of image"""
        return np.array(self.size_volume)

    @property
    def size_im_enlarge(self):
        """Size of enlarged image"""
        if self._size_im_enlarge is None:
            extra_sz = (150, 200, 100)
            self._size_im_enlarge = np.array(self.size_volume) + np.array(extra_sz)
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

    @classmethod
    def from_json(cls, json_file: str) -> list['RayCellParams']:
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
    def from_dict(cls, data: dict) -> 'RayCellParams':
        """Create an instance from a JSON file"""
        data = {cls.params_map.get(k, k): v for k, v in data.items()}
        return cls(**data)
