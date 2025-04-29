"""Spruce microstructure generation module."""
import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from . import ray_cells as rcl
from . import vessels as ves
from .clocks import Clock
from .microstructure import WoodMicrostructure


class SpruceMicrostructure(WoodMicrostructure):
    save_prefix = 'SaveSpruce'
    local_distortion_cutoff = 200
    ray_height_mod = 5
    skip_cell_thick_rescale = 1.5

    def get_distortion_map(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Generate the distortion map for early wood and late wood"""
        pparam = self.params.period_parameter
        cell_r = self.params.cell_r
        sie_x = self.params.size_im_enlarge[0]

        ind_start = 2 * pparam  # This value is not important. It should be roughly this value
        lim = ind_start + sie_x
        thicker_all = np.empty((0,))
        compress_all = np.empty((0,))
        temp = 0

        while len(thicker_all) < lim:
            feat_size = np.round(pparam * (1 + np.random.rand()))  # The size of a year ring is roughly controlled by this
            dx = np.arange(np.round(0.4 * feat_size), feat_size + 1)
            thick_amplitude = cell_r / 2 * (1 + 0.5 * np.random.rand())  # Control the amplitude of the cell wall thickness.
            # They are added to the original one
            thicker1 = thick_amplitude / feat_size**4 * dx**4
            thicker1 -= thicker1[0]
            thicker_all = np.concatenate((thicker_all, thicker1), axis=0)
            # thicker_all.append(thicker1)

            # The fiber size is different in early wood and late wood. Their size
            # can be controlled by applying a compression to the whole field. The
            # compression is designed
            k = cell_r / 1.5 / 3 * feat_size / 14.5
            compress = -k / feat_size**3 * dx**3
            compress += temp - compress[0]
            compress_all = np.concatenate((compress_all, compress), axis=0)
            temp = compress_all[-1]

        # print('thicker_all.shape: ', thicker_all.shape)
        thick_all_valid_sub = thicker_all[ind_start:ind_start + sie_x]
        compress_all_valid_sub = compress_all[ind_start:ind_start + sie_x]
        compress_all_valid_sub -= (
            compress_all_valid_sub[0] +
            (compress_all_valid_sub[-1] - compress_all_valid_sub[0]) * np.arange(sie_x) / sie_x
        )

        return thick_all_valid_sub, compress_all_valid_sub

    def get_grid_all(self, thick_all_valid_sub: npt.NDArray):
        """Specify the location of grid nodes and the thickness (with disturbance)"""
        gx, gy = self.params.x_grid.shape
        gz = self.params.size_im_enlarge[2]
        ds = self.params.slice_interest_space

        cwt = self.params.cell_wall_thick

        slice_interest = np.arange(0, gz, ds)
        l = len(slice_interest)

        x_grid_interp = np.random.rand(gx, gy, l) * 3 - 1.5 + self.params.x_grid[..., np.newaxis]
        y_grid_interp = np.random.rand(gx, gy, l) * 3 - 1.5 + self.params.y_grid[..., np.newaxis]
        thickness_interp = thick_all_valid_sub[np.round(x_grid_interp).astype(int)]
        thickness_interp_ray = (cwt - 0.5) * np.ones((gx, gy, l)) + np.random.rand(gx, gy, l)
        thickness_interp_fiber = thickness_interp_ray + thickness_interp

        interp_z = np.arange(gz)
        x_grid_all = np.empty((gx, gy, gz))
        y_grid_all = np.empty_like(x_grid_all)
        thickness_all_ray = np.empty_like(x_grid_all)
        thickness_all_fiber = np.empty_like(x_grid_all)
        for i in range(gx):
            for j in range(gy):
                x_grid_all[i, j, :] = CubicSpline(slice_interest, x_grid_interp[i, j, :])(interp_z)
                y_grid_all[i, j, :] = CubicSpline(slice_interest, y_grid_interp[i, j, :])(interp_z)
                thickness_all_ray[i, j, :] = CubicSpline(slice_interest, thickness_interp_ray[i, j, :])(interp_z)
                thickness_all_fiber[i, j, :] = CubicSpline(slice_interest, thickness_interp_fiber[i, j, :])(interp_z)

        self.x_grid_all = x_grid_all[..., self.params.save_slice]
        self.y_grid_all = y_grid_all[..., self.params.save_slice]
        self.thickness_all_ray = thickness_all_ray[..., self.params.save_slice]
        self.thickness_all_fiber = thickness_all_fiber[..., self.params.save_slice]

        return x_grid_all, y_grid_all, thickness_all_ray, thickness_all_fiber

    def get_ray_cell_indexes(self) -> npt.NDArray:
        """Get ray cell indexes"""
        ly = len(self.params.y_vector)
        ray_cell_x_ind_all = np.empty((1, 0))
        if self.params.is_exist_ray_cell:
            ray_cell_x_ind_all = rcl.get_x_indexes(ly, 8, np.round((ly - 15) / 10), 6)
        return ray_cell_x_ind_all.astype(int)

    @Clock('vessels')
    def generate_vessel_indexes(self, ray_cell_x_ind_all: npt.NDArray = None):
        """Get vessels"""
        self.logger.info('=' * 80)
        self.logger.info('Generating vessels...')
        if not self.params.is_exist_vessel:
            return np.empty((0, 2), dtype=int)
        lx = len(self.params.x_vector)
        ly = len(self.params.y_vector)

        vc1 = 60
        vc2 = 50

        vessel_all = ves.generate_indexes(vc1, vc2, lx, ly)

        return vessel_all.astype(int)

    def get_indx_skip_all(self, vessel_all: npt.NDArray) -> npt.NDArray:
        """Get the indexes of the grid nodes where fibers are not generated"""
        return np.empty((0, 6, 2), dtype=int)

    def get_indx_ves_edges(self, vessel_all: npt.NDArray) -> npt.NDArray:
        """Get the indexes of the grid nodes at the edges of the vessels"""
        return np.empty((0, 6, 2), dtype=int)

    def get_indx_vessel_cen(self, vessel_all: npt.NDArray) -> npt.NDArray:
        """Get the indexes of the grid nodes where fibers are not generated"""
        return np.empty((0, 2), dtype=int)

    def _get_small_fiber_exp(self, is_close_to_ray: npt.NDArray) -> npt.NDArray:
        """Get the exponent for small fiber generation"""
        lx, ly = is_close_to_ray.shape
        exp_ellipse_2 = 5 + np.round(np.random.rand(lx, ly) * 2)  # Internal contour
        mask = np.random.rand(lx, ly) < 0.8
        exp_ellipse_2[~is_close_to_ray & mask] -= 2
        return exp_ellipse_2

    def generate_large_fibers(
            self,
            indx_vessel: npt.NDArray,
            indx_vessel_cen: npt.NDArray,
            # indx_skip_all: npt.NDArray,
            input_volume: npt.NDArray
        ) -> npt.NDArray:
        """Generate large fibers."""
        return input_volume

    def _generate_raycell_cell_r(self, interp1: npt.NDArray, interp2: npt.NDArray, dx: npt.NDArray, k: int):
        """Get the value of `cell_r` for `generate_raycell`"""
        ray_height = self.params.ray_height
        sie_z = self.params.size_im_enlarge[2]
        return np.column_stack((
            (interp2 - interp1) / 2,
            np.full(dx.shape, (np.min((k + ray_height, sie_z)) - np.max((1, k))) / 2)
        ))
    def _generate_raycell_valid_idx(
            self, vel_col_r: npt.NDArray, vel_col_r1: npt.NDArray, flag: int, cet: int
        ) -> npt.NDArray:
        """Get the value of `valid_idx` for `generate_raycell`

        Args:
            vel_col_r (npt.NDArray): First column of the ray cell
            vel_col_r1 (npt.NDArray): Second column of the ray cell
            flag (int): -1/+1 if first/last ray cell, 0 otherwise
            cet (int): Cell end thickness
        """
        if flag == -1:
            start = cet
            end = -cet // 2
        elif flag == 1:
            start = cet // 2 + 1
            end = -cet
        else:
            start = cet // 2 + 1
            end = -cet // 2

        # -1 for 0-indexing
        valid_idx = np.arange(
            vel_col_r + start - 1,
            vel_col_r1 + end
        )
        valid_idx = set(int(_) for _ in valid_idx)
        return valid_idx

    def _get_k_grid1(self, is_ctr: npt.NDArray, is_ctr_far: npt.NDArray, vess_cond: npt.NDArray) -> npt.NDArray:
        """Get the k_grid of parameters for the deformation map"""
        lx, ly = is_ctr.shape

        k_grid = np.empty((lx, ly, 4))

        k_grid[..., 0] = 0.1
        k_grid[..., 1] = 0.08
        k_grid[..., 2] = 2
        k_grid[..., 3] = 8 + np.random.rand(lx, ly) * 10

        return k_grid

    def _get_k_grid2(
        self, k_grid: npt.NDArray, is_ctr: npt.NDArray, is_ctr_far: npt.NDArray, vess_cond: npt.NDArray
    ) -> npt.NDArray:
        """Regenerate part of the k_grid for different random numbers between U and V computation."""
        lx, ly = is_ctr.shape

        k_grid[is_ctr, 3] = 2 + np.random.rand(lx, ly)[is_ctr] * 2
        k_grid[~is_ctr, 3] = 6 + np.random.rand(lx, ly)[~is_ctr] * 9

        return k_grid

    def _get_sign_grid(self, vess_cond):
        """Get the sign grid of parameters for accumulating the deformation map"""
        lx, ly = vess_cond.shape
        s_grid = np.ones((lx, ly))

        return s_grid
