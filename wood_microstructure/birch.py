"""Birch microstructure generation module."""
import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from . import distortion as dist
from . import ray_cells as rcl
from . import vessels as ves
from .clocks import Clock
from .microstructure import WoodMicrostructure
from .params import BirchParams


class BirchMicrostructure(WoodMicrostructure):
    ParamsClass = BirchParams

    save_prefix = 'SaveBirch'
    local_distortion_cutoff = 200
    ray_height_mod = 6
    skip_cell_thick_rescale = 2.0
    model_commit = 'aef50579790849ba9c30a91a16688921aba9ac19'

    def get_distortion_map(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Generate the distortion map for early wood and late wood"""
        return np.empty((0, )), np.empty((0, ))

    def get_grid_all(self, thick_all_valid_sub: npt.NDArray):
        """Specify the location of grid nodes and the thickness (with disturbance)"""
        gx, gy = self.params.x_grid.shape
        gz = self.params.size_im_enlarge[2]

        slice_interest = self.slice_interest
        l = len(slice_interest)

        x_grid_interp = np.random.rand(gx, gy, l) * 3 - 1.5 + self.params.x_grid[..., np.newaxis]
        y_grid_interp = np.random.rand(gx, gy, l) * 3 - 1.5 + self.params.y_grid[..., np.newaxis]
        thickness_interp = np.random.rand(gx, gy, l) + self.params.cell_wall_thick - 0.5

        interp_z = np.arange(gz)
        x_grid_all = np.empty((gx, gy, gz))
        y_grid_all = np.empty_like(x_grid_all)
        thickness_all = np.empty_like(x_grid_all)
        for i in range(gx):
            for j in range(gy):
                x_grid_all[i, j, :] = CubicSpline(slice_interest, x_grid_interp[i, j, :])(interp_z)
                y_grid_all[i, j, :] = CubicSpline(slice_interest, y_grid_interp[i, j, :])(interp_z)
                thickness_all[i, j, :] = CubicSpline(slice_interest, thickness_interp[i, j, :])(interp_z)

        self.x_grid_all = x_grid_all[..., self.params.save_slice]
        self.y_grid_all = y_grid_all[..., self.params.save_slice]
        self.thickness_all_fiber = thickness_all[..., self.params.save_slice]
        self.thickness_all_ray = thickness_all[..., self.params.save_slice]

        return x_grid_all, y_grid_all, thickness_all, thickness_all

    @Clock.register('rcl:indexes')
    def get_ray_cell_indexes(self) -> npt.NDArray:
        """Get ray cell indexes"""
        ly = len(self.params.y_vector)
        ray_cell_x_ind_all = np.empty((1, 0))
        if self.params.is_exist_ray_cell:
            ray_cell_x_ind_all = rcl.get_x_indexes(ly, 10, self.params.ray_space, 10)
        return ray_cell_x_ind_all.astype(int)

    @Clock.register('vessels')
    def generate_vessel_indexes(self, ray_cell_x_ind_all: npt.NDArray = None) -> npt.NDArray:
        """Get vessels"""
        self.logger.info('=' * 80)
        self.logger.info('Generating vessels...')
        if not self.params.is_exist_vessel:
            return np.empty((0, 2), dtype=int)
        lx = len(self.params.x_vector)
        ly = len(self.params.y_vector)

        vc1 = self.params.vessel_count
        vc2 = self.params.vessel_count // 2

        vessel_all = ves.generate_indexes(vc1, vc2, lx, ly)
        self.logger.debug('  -- Vessel filter close --')
        vessel_all = ves.filter_close(vessel_all)
        self.logger.debug('  -- Vessel filter ray close --')
        vessel_all = ves.filter_ray_close(vessel_all, ray_cell_x_ind_all)
        self.logger.debug('  -- Vessel extend --')
        vessel_all = ves.extend(vessel_all, lx, ly)
        # vessel_all = self.vessel_filter_close(vessel_all)
        self.logger.debug('  -- Vessel filter ray close --')
        vessel_all = ves.filter_ray_close(vessel_all, ray_cell_x_ind_all)
        self.logger.debug('  -- Vessel filter edge --')
        vessel_all = ves.filter_edge(vessel_all, lx, ly)

        return vessel_all.astype(int)

    def get_indx_skip_all(self, vessel_all: npt.NDArray) -> npt.NDArray:
        """Get the indexes of the grid nodes where fibers are not generated"""
        return ves.get_grid_idx_in_vessel(vessel_all)

    def get_indx_ves_edges(self, vessel_all: npt.NDArray) -> npt.NDArray:
        """Get the indexes of the grid nodes at the edges of the vessels"""
        return ves.get_grid_idx_edges(vessel_all)

    def get_indx_vessel_cen(self, vessel_all: npt.NDArray) -> npt.NDArray:
        """Get the indexes of the grid nodes where fibers are not generated"""
        return vessel_all

    def _get_small_fiber_exp(self, is_close_to_ray: npt.NDArray) -> npt.NDArray:
        """Get the exponent for small fiber generation"""
        lx, ly = is_close_to_ray.shape
        exp_ellipse_2 = np.full((lx, ly), 2, dtype=int)
        return exp_ellipse_2

    def _generate_raycell_cell_r(self, interp1: npt.NDArray, interp2: npt.NDArray, dx: npt.NDArray, k: int):
        """Get the value of `cell_r` for `generate_raycell`"""
        ray_height = self.params.ray_height
        sie_z = self.params.size_im_enlarge[2]
        return np.column_stack((
            (interp2 - interp1) / 2,
            np.full(dx.shape, (np.min((k + ray_height, sie_z)) - np.max((1, k))) / 2)
        )) + 0.5

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
            start = cet // 2 - 1
            end = -cet // 2
        else:
            start = cet // 2 - 1
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

        k_grid[..., 0] = 0.08
        k_grid[..., 1] = 0.06
        k_grid[..., 2] = 2
        k_grid[..., 3] = 15

        k_grid[~vess_cond, 2] = 2 + np.random.rand(lx, ly)[~vess_cond]
        k_grid[~vess_cond, 3] = 1 + np.random.rand(lx, ly)[~vess_cond]
        k_grid[~vess_cond & is_ctr, 3] *= 0.3

        k_grid[vess_cond, 0] = 0.06
        k_grid[vess_cond, 1] = 0.055
        # k_grid[vess_cond, 2] = 2
        k_grid[vess_cond & is_ctr_far, 3] = 3 + 5 * np.random.rand(lx, ly)[vess_cond & is_ctr_far]

        return k_grid

    def _get_k_grid2(
        self, k_grid: npt.NDArray, is_ctr: npt.NDArray, is_ctr_far: npt.NDArray, vess_cond: npt.NDArray
    ) -> npt.NDArray:
        """Regenerate part of the k_grid for different random numbers between U and V computation."""
        lx, ly = is_ctr.shape

        k_grid[~vess_cond, 2] = 2 + np.random.rand(lx, ly)[~vess_cond]
        k_grid[~vess_cond, 3] = 1 + np.random.rand(lx, ly)[~vess_cond]
        k_grid[~vess_cond & is_ctr, 3] *= 0.3

        return k_grid

    def _get_sign_grid(self, vess_cond):
        """Get the sign grid of parameters for accumulating the deformation map"""
        lx, ly = vess_cond.shape

        s_grid = np.sign(np.random.randn(lx, ly))
        s_grid[vess_cond] = -1

        return s_grid

    def _get_u1_v1(self, xc_grid, yc_grid, is_close_to_ray_far, sie_x, sie_y):
        """Get the local distortion map u1, v1"""
        u1 = np.zeros((sie_x, sie_y), dtype=float)
        v1 = np.zeros((sie_x, sie_y), dtype=float)


        for xc, yc, cf in zip(xc_grid.flatten(), yc_grid.flatten(), is_close_to_ray_far.flatten()):
            if np.random.rand() >= 1 / 100:
                continue
            app = 0.2 if cf else 0.5
            k = [0.01, 0.008, 1.5 * (1 + np.random.rand()), app * (1 + np.random.rand())]

            xp, yp = dist.get_distortion_grid(
                xc, yc, sie_x, sie_y,
                # This deformation is much longer range so it needs to be applied to the entire area
                max(sie_x, sie_y)
                # self.local_distortion_cutoff
            )
            if np.random.randn() > 0:
                local_dist = dist.local_distort(xp, yp, xc, yc, k)
                u1[xp, yp] += np.sign(np.random.randn()) * local_dist
            else:
                local_dist = dist.local_distort(yp, xp, yc, xc, k)
                v1[xp, yp] += np.sign(np.random.randn()) * local_dist

        return u1, v1

    def _get_global_interp_grid(
            self,
            x_grid: npt.NDArray, y_grid: npt.NDArray, z_grid: npt.NDArray,
            u1: npt.NDArray, v1: npt.NDArray
        ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Get the interpolation grid for global deformation"""
        sie_x, sie_y, _ = self.params.size_im_enlarge

        # v_allz   = (x_grid-sizeImEnlarge(1)/3).^2/1e4/8+(y_grid-sizeImEnlarge(2)/2).*(x_grid-sizeImEnlarge(1)/3)/1e4/9;
        v_all_z = (
            (x_grid - sie_x / 3)**2 / 1e4 / 8 +
            (y_grid - sie_y / 2) * (x_grid - sie_x / 3) / 1e4 / 9
        ) + v1[..., np.newaxis]
        # u_allz   = (y_grid-sizeImEnlarge(1)/3).^2/1e4/9+(x_grid-sizeImEnlarge(2)/2).*(y_grid-sizeImEnlarge(1)/3)/1e4/8;
        u_all_z = (
            (y_grid - sie_x / 3)**2 / 1e4 / 9 +
            (x_grid - sie_y / 2) * (y_grid - sie_x / 3) / 1e4 / 8
        ) + u1[..., np.newaxis]

        x_interp = x_grid - u_all_z
        y_interp = y_grid - v_all_z
        z_interp = z_grid


        return x_interp, y_interp, z_interp, u_all_z, v_all_z
