"""Functions to generate/handle ray cells in the wood microstructure."""
import os
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.interpolate import CubicSpline, griddata

from . import distortion as dist
from . import vessels as ves
from .clocks import Clock
# from .distortion import get_distortion_grid, local_distort
from .fit_elipse import fit_elipse, fit_ellipse_6pt
from .loggers import get_logger
from .params import BirchParams

logger = get_logger()

class RayCellSplineError(Exception):
    """Exception raised when the spline interpolation fails."""

@Clock('ray_cell')
@Clock('rcl:indexes')
def get_x_indexes(ly: int, trim: int, ray_space: float, random_width: float = 10) -> npt.NDArray:
    """Get the x indexes of the ray cells.

    Args:
        ly (int): Number of grid nodes in y direction
        trim (int): Trim `trim` number of grid nodes from the start and end
        ray_space (float): Distance between ray cells
        space (float): Space between ray cells
        random_width (float): Randomiazion of the ray cell width (+/- random_width / 2)

    Returns:
        npt.NDArray: X indexes of the ray cells
    """
    rwh = random_width / 2
    ray_cell_linspace = np.arange(trim - 1, ly - trim, ray_space)
    indexes = ray_cell_linspace + np.random.rand(len(ray_cell_linspace)) * random_width - rwh
    indexes = np.floor(indexes / 2) * 2

    return indexes.astype(int)

@Clock('ray_cell')
@Clock('rcl:distribute')
def distribute(
        sie_z: int, indexes: npt.NDArray, cell_num: float, cell_num_std: float, height: float,
        height_mod: int
    ) -> tuple[npt.NDArray, list[npt.NDArray], npt.NDArray]:
    """Distribute the ray cells across the volume

    Args:
        sie_z (int): Size of the volume in z direction
        indexes (npt.NDArray): Ray cell indices
        cell_num (float): number of cells
        cell_num_std (float): standard deviation of the number of cells
        height (float): height of the ray cells
        height_mod (int): height modifier

    Returns:
        tuple[npt.NDArray, list[npt.NDArray]]:
        - Ray cell indices (num_ray_cells, 2): indices array A where A[j][1] = A[j][0] + 1
        - Ray cell widths (num_ray_cells, non_uniform): length of elements depends on the randomly generated group
    """
    x_ind = []
    width = []

    m = int(np.ceil(sie_z / cell_num / height + 6))
    for idx in indexes:
        app = [0]
        ray_cell_space = np.round(16 * np.random.rand(m)) + height_mod
        rnd = np.round(-30 * np.random.rand())
        # ray_idx = [idx, idx + 1]
        for rs in ray_cell_space:
            group = np.random.randn() * cell_num_std + cell_num
            group = np.clip(group, 5, 25)
            app = app[-1] + (np.arange(group + 1) + rs + rnd) * height
            rnd = 0

            if app[0] > sie_z - 150:
                break

            if app[-1] >= 150:  # TODO should 150 be a parameter indicating the start of the ray cell?
                # x_ind.append(ray_idx)
                x_ind.append(idx)
                width.append(np.round(app).astype(int))

    return np.array(x_ind, dtype=int), width


# def get_vessel_end_loc(sie_x: int, length: float, variance: float, shape: tuple | int = None) -> npt.NDArray:
#     """Generate the vessel end location

#     Args:
#         sie_x (int): Size of the volume in x direction
#         length (float): Length of the ray cell
#         variance (float): Variance of the ray cell length
#         shape (tuple | int, optional): Generate end locations for a grid of shape. Defaults to None = 0D-grid.

#     Returns:
#         npt.NDArray: Array of vessel end locations of shape (shape, N) where N depends on random generation
#     """

#     if shape is None:
#         shape = tuple()
#     elif isinstance(shape, int):
#         shape = (shape,)

#     # ray_height = self.params.ray_height
#     rcl_d3 = length / 3
#     rcl_t2 = length * 2

#     lim = sie_x + length

#     vessel_end_loc = np.random.rand(*shape, 1) * length
#     while np.any(vessel_end_loc[...,-1] < lim):
#         tmp = length + variance * np.random.randn(*shape, 1)
#         tmp[tmp < rcl_d3] = rcl_t2
#         tmp[tmp > rcl_t2] = rcl_t2
#         vessel_end_loc = np.concatenate((vessel_end_loc, vessel_end_loc[..., -1:] + tmp), axis=-1)
#     vessel_end_loc = np.round(vessel_end_loc)

#     return vessel_end_loc.astype(int)

# @Clock('ray_cell')
# @Clock('rcl:generate')
# def generate(
#         sie_x: int,
#         sie_z: int,
#         x_grid_all: npt.NDArray,
#         y_grid_all: npt.NDArray,
#         thick_all: npt.NDArray,
#         length: float,
#         variance: float,
#         height: float,
#         end_thick: float,
#         ray_idx: tuple[int, int],
#         ray_width: npt.NDArray,
#         save_slices: list[int],
#         vol_img: npt.NDArray
#     ) -> npt.NDArray:
#     """Generate ray cell

#     Args:
#         ray_idx (tuple[int, int]): Tuple of (idx, idx+1) where idx is the y-index of the ray cell
#         ray_width (npt.NDArray): Ray cell width
#         input_volume (npt.NDArray): Input 3D gray-scale image volume to modify

#     Returns:
#         npt.NDArray: Modified 3D gray-scale image volume with ray cells
#     """
#     rcl_d2 = length / 2
#     rcl_d2_r = np.round(length / 2)

#     cet_d2_p1 = end_thick // 2 + 1
#     cet_d2_m1 = end_thick // 2 - 1

#     ray_idx = np.array(ray_idx).flatten().astype(int)
#     ray_width = np.array(ray_width).flatten().astype(int)

#     dx = np.arange(sie_x)

#     vessel_end_loc = get_vessel_end_loc(sie_x, length, variance)

#     ray_column_rand = int(np.round(1 / 2 * height))

#     for m2, j_slice in enumerate(ray_width):
#         # self.logger.debug('  %d/%d   %d', m2, len(ray_width), j_slice)
#         k0 = j_slice
#         k1 = j_slice + ray_column_rand
#         tmp0_1 = (max(1, k0) + min(k0 + height, sie_z)) / 2
#         tmp1_1 = (max(1, k1) + min(k1 + height, sie_z)) / 2
#         # tmp_1 = (np.max((1, k)) + np.min((k + ray_height, sie_z))) / 2

#         t0 = int(np.round(tmp0_1)) - 1  # 0-indexed
#         t1 = int(np.round(tmp1_1)) - 1

#         for i, column_idx in enumerate(ray_idx):
#             # self.logger.debug('Ray cell: %d %s', column_idx, ray_idx)
#             # vessel_end_loc = self.get_vessel_end_loc().reshape(-1)

#             if column_idx % 2:
#                 t, k, tmp_1 = t1, k1, tmp1_1
#             else:
#                 t, k, tmp_1 = t0, k0, tmp0_1

#             if t < 0 or t >= sie_z - 11:
#                 continue
#             if t not in save_slices:
#                 continue


#             vel = vessel_end_loc[i]
#             vel = vel[vel <= sie_x + rcl_d2]

#             # rand = ray_column_rand * ((column_idx + 1) % 2)

#             interp_x0 = x_grid_all[:, column_idx, t]
#             interp_y0 = y_grid_all[:, column_idx, t]
#             interp_x1 = x_grid_all[:, column_idx + 1, t]
#             interp_y1 = y_grid_all[:, column_idx + 1, t]
#             interp_thick = thick_all[:, column_idx, t]
#             tmp_2 = rcl_d2_r if m2 % 2 else 0

#             try:
#                 y_interp1_c = CubicSpline(interp_x0, interp_y0)(dx) - 1.5
#                 y_interp2_c = CubicSpline(interp_x1, interp_y1)(dx) + 1.5
#                 thick_interp_c = CubicSpline(interp_x0, interp_thick)(dx)
#             except ValueError:
#                 # TODO: check how matlab handles this
#                 # The spline interpolations as is now can generate an X/Y grid where 2 points can swap order
#                 # e.g. on x-coords the left points can get pushed right enough and the right-point left enough
#                 # for them to swap order causing successive spline interpolation to fail because the x-coords
#                 # are not monotonic
#                 self.logger.warning('    WARNING: Spline interpolation failed')

#                 continue


#             cell_center = np.column_stack((
#                 dx,
#                 np.round((y_interp2_c + y_interp1_c)) / 2,
#                 np.full(dx.shape, tmp_1)
#             ))
#             cell_r = np.column_stack((
#                 (y_interp2_c - y_interp1_c) / 2,
#                 np.full(dx.shape, (np.min((k + height, sie_z)) - np.max((1, k))) / 2)
#             )) + 0.5

#             d_col = int(end_thick)
#             for vel_r, vel_r1 in zip(vel[:-1], vel[1:]):
#                 # tmp_2 = rcl_d2_r if m2 % 2 else 0
#                 vel_col_r = int(vel_r + tmp_2)
#                 if vel_col_r < 1:
#                     continue
#                 vel_col_r1 = int(vel_r1 + tmp_2)
#                 if vel_col_r1 > sie_x - 1:
#                     continue
#                 cell_neigh_pt = np.array([
#                     [vel_col_r, vel_col_r1],
#                     [np.round(y_interp1_c[vel_col_r]), np.round(y_interp2_c[vel_col_r])],
#                     [np.max((0, k)), np.min((k + height, sie_x))]
#                 ])

#                 valid_idx = np.arange(
#                     vel_col_r + d_col,
#                     vel_col_r1 - cet_d2_p1  #Right inclusive
#                 )
#                 valid_idx = set(int(_) for _ in valid_idx)
#                 d_col = cet_d2_m1

#                 for idx in range(vel_col_r, vel_col_r1):
#                     if j_slice == np.min(ray_width):
#                         vol_img[
#                             idx,
#                             int(y_interp1_c[idx]):int(y_interp2_c[idx] + 1),
#                             int(cell_center[idx, 2]):int(cell_neigh_pt[2, 1] + 1)
#                         ] = 255
#                     elif j_slice == np.max(ray_width):
#                         vol_img[
#                             idx,
#                             int(y_interp1_c[idx]):int(y_interp2_c[idx] + 1),
#                             int(cell_neigh_pt[2, 0]):int(cell_center[idx, 2])
#                         ] = 255
#                     else:
#                         vol_img[
#                             idx,
#                             int(y_interp1_c[idx]):int(y_interp2_c[idx] + 1),
#                             int(cell_neigh_pt[2, 0]):int(cell_neigh_pt[2, 1])
#                         ] = 255

#                     if idx not in valid_idx:
#                         continue

#                     for j in range(int(cell_neigh_pt[1, 0]), int(cell_neigh_pt[1, 1]) + 1):
#                         for s in range(int(cell_neigh_pt[2, 0]), int(cell_neigh_pt[2, 1]) + 1):
#                             outer_elipse = (
#                                 (j - cell_center[idx, 1])**2 / cell_r[idx, 0]**2 +
#                                 (s - cell_center[idx, 2])**2 / cell_r[idx, 1]**2
#                             )

#                             if outer_elipse < 1:
#                                 inner_elipse = (
#                                     (j - cell_center[idx, 1])**2 / (cell_r[idx, 0] - thick_interp_c[idx])**2 +
#                                     (s - cell_center[idx, 2])**2 / (cell_r[idx, 1] - thick_interp_c[idx])**2
#                                 )
#                                 # print('   ', j, s)
#                                 vol_img[idx, j, s] = int(
#                                     (1 / (1 + np.exp(-(inner_elipse - 1) / .05))) * 255
#                                 )

#     return vol_img
