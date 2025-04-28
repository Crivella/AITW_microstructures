"""Ray cells"""
import os
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.interpolate import CubicSpline, griddata

from . import distortion as dist
from . import ray_cells as rcl
from .clocks import Clock
# from .distortion import get_distortion_grid, local_distort
from .fit_elipse import fit_ellipse_6pt
from .loggers import add_file_logger, get_logger
from .params import BaseParams


class WoodMicrostructure(ABC):
    """Base class for wood microstructure generation"""
    @property
    @abstractmethod
    def local_distortion_cutoff(self) -> int:
        """Local distortion cutoff value"""

    @property
    @abstractmethod
    def ray_height_mod(self) -> int:
        """Ray height modification value used for ray cell distribution"""

    @property
    @abstractmethod
    def save_prefix(self) -> int:
        """Local distortion cutoff value"""

    def __init__(self, params: BaseParams, outdir: str = None):
        self.params = params

        self.x_grid_all = None
        self.y_grid_all = None

        self.outdir = outdir or os.getenv('ROOT_DIR', '.')

        self.get_root_dir()
        log_file = os.path.join(self.root_dir, 'wood_microstructure.log')
        self.logger = get_logger()
        add_file_logger(self.logger, log_file)

        save_param_file = os.path.join(self.root_dir, 'params.json')
        self.params.to_json(save_param_file)

    def get_root_dir(self) -> str:
        """Get the root directory for saving files"""
        dir_cnt = 0
        while os.path.exists(os.path.join(self.outdir, f'{self.save_prefix}_{dir_cnt}')):
            dir_cnt += 1
        while True:
            try:
                dir_path = os.path.join(self.outdir, f'{self.save_prefix}_{dir_cnt}')
                os.makedirs(dir_path)
            except FileExistsError:
                dir_cnt += 1
                continue
            else:
                self.root_dir = dir_path
                break

    def distrbute_ray_cells(self, ray_cell_x_ind_all: npt.NDArray) -> tuple[
            npt.NDArray,
            list[npt.NDArray],
            npt.NDArray
        ]:
        """Distribute the ray cells across the volume

        Args:
            ray_cell_x_ind_all (npt.NDArray): Ray cell  indices

        Returns:
            tuple[npt.NDArray, list[npt.NDArray]]:
            - Ray cell indices (num_ray_cells,): Array of indices of the ray cells (without the +1 column)
            - Ray cell widths (num_ray_cells, non_uniform): length of elements depends on the randomly generated group
        """
        self.logger.info('=' * 80)
        self.logger.info('Distributing ray cells...')

        sie_z = self.params.size_im_enlarge[2]
        ray_cell_num = self.params.ray_cell_num
        ray_cell_num_std = self.params.ray_cell_num_std
        ray_height = self.params.ray_height

        return rcl.distribute(
            sie_z, ray_cell_x_ind_all, ray_cell_num, ray_cell_num_std, ray_height,
            height_mod = self.ray_height_mod,
        )

    def get_fiber_end_condition(self, lx: int, ly: int, i_slice: int) -> npt.NDArray:
        """Get a condition for skipping fiber generation due to fiber ending"""
        sie_z = self.params.size_im_enlarge[2]

        cell_length = self.params.cell_length
        cell_length_variance = self.params.cell_length_variance
        cell_end_thick = self.params.cell_end_thick

        num_fiber_end_loc = int(np.ceil(sie_z / cell_length)) + 7

        fiber_end = np.random.rand(lx, ly, 1) * cell_length
        for _ in range(num_fiber_end_loc):
            tmp = np.clip(cell_length + np.random.randn(lx, ly) * cell_length_variance, 100, 3*cell_length)
            fiber_end = np.concatenate((fiber_end, fiber_end[:, :, -1:] + tmp[..., np.newaxis]), axis=-1)
        fiber_end = np.round(fiber_end)
        # print('fiber_end_loc_all:', fiber_end_loc_all.shape)

        # This is a manually given value. To increase the randomness
        fiber_end = fiber_end - 4 * cell_length + 1
        # The end of the vessel should be inside the volume.
        fiber_end[(fiber_end < 4) | (fiber_end > sie_z - 4)] = np.nan

        ct = np.arange(int(cell_end_thick))
        fiber_end = fiber_end[..., np.newaxis] + ct[np.newaxis, np.newaxis, np.newaxis, :]
        fiber_end = fiber_end.reshape(lx, ly, -1).astype(int)
        fiber_end_cond = np.any(fiber_end == i_slice, axis=-1)

        return fiber_end_cond

    @Clock('large_fibers')
    def generate_large_fibers(
            self,
            indx_vessel: npt.NDArray,
            indx_vessel_cen: npt.NDArray,
            # indx_skip_all: npt.NDArray,
            input_volume: npt.NDArray
        ) -> npt.NDArray:
        """Generate large fibers."""
        self.logger.info('=' * 80)
        self.logger.info('Generating large fibers...')
        self.logger.debug('  indx_vessel: %s', indx_vessel.shape)
        self.logger.debug('  indx_vessel_cen: %s', indx_vessel_cen.shape)
        vol_img_ref = np.copy(input_volume)

        x_vector = self.params.x_vector
        y_vector = self.params.y_vector

        sie_x, sie_y, sie_z = self.params.size_im_enlarge

        # vessel_length = self.params.vessel_length
        vessel_thicker = self.params.vessel_thicker
        # vessel_length_variance = self.params.vessel_length_variance
        # cell_end_thick = self.params.cell_end_thick

        x_grid_all = self.x_grid_all
        y_grid_all = self.y_grid_all

        for i in range(1, len(x_vector)-2, 2):
            for j in range(1, len(y_vector)-2, 2):
                # The arrangement of the cells should be staggered. So every four nodes,
                # they should deviate along x direction.
                i1 = i + ((j + 1) % 4 == 0)
                i_vessel = np.where(np.all(indx_vessel_cen == (i1, j), axis=1))[0]
                if not i_vessel.size:
                    continue
                if len(i_vessel) > 1:
                    raise ValueError('More than one vessel in the same cell')
                self.logger.debug('  large vessel at: idx = (%s, %s)', i1, j)
                i_vessel = i_vessel[0]
                six_pt_x = indx_vessel[i_vessel, :, 0]
                six_pt_y = indx_vessel[i_vessel, :, 1]

                # vessel_end_loc_all = [np.round(np.random.rand() * vessel_length)]
                # for _ in range(int(np.ceil(sie_z / vessel_length)) + 8):
                #     # temp = np.min(
                #     #     3 * vessel_length,
                #     #     np.max(100, vessel_length + np.random.randn() * self.params.vessel_length_variance)
                #     # )
                #     temp = np.clip(vessel_length + np.random.randn() * vessel_length_variance, 100, 3 * vessel_length)
                #     vessel_end_loc_all.append(np.round(vessel_end_loc_all[-1] + temp))
                # vessel_end_loc_all = np.array(vessel_end_loc_all)

                # # This is a manually given value. To increase the randomness
                # vessel_end_loc = vessel_end_loc_all - 4 * vessel_length + 1
                # # The end of the vessel should be inside the volume.
                # vessel_end_loc = vessel_end_loc[(vessel_end_loc >= 4) & (vessel_end_loc <= sie_z - 4)]

                # vessel_end = np.empty((0, vessel_end_loc.size))
                # for ct in range(cell_end_thick):
                #     vessel_end = np.vstack((vessel_end, vessel_end_loc + ct))


                # for i_slice in range(sie_z):
                for i_slice in self.params.save_slice:
                    point_coord = np.column_stack((
                        x_grid_all[six_pt_x, six_pt_y, i_slice],
                        y_grid_all[six_pt_x, six_pt_y, i_slice]
                    ))
                    r1, r2, h, k = fit_ellipse_6pt(point_coord)  # Estimate the coefficients of the ellipse.

                    thick = self.thickness_all[i1, j, i_slice] + vessel_thicker
                    mr = np.floor(max(r1, r2))

                    x_grid, y_grid = np.mgrid[
                        max(0, int(np.ceil(h)) - mr):min(sie_x, int(np.ceil(h)) + mr),
                        max(0, int(np.ceil(k)) - mr):min(sie_y, int(np.ceil(k)) + mr)
                    ].astype(int)
                    in_elipse1 = (
                        (x_grid - h)**2 / (r1 - thick * 4/3)**2 +
                        (y_grid - k)**2 / (r2 - thick * 1)**2
                    )
                    in_elipse2 = (
                        (x_grid - h)**2 / (r1 - thick * 5/3)**2 +
                        (y_grid - k)**2 / (r2 - thick * 5/3)**2
                    )
                    mul = 1 + np.exp(-(in_elipse2 - 1) / 0.05)
                    cond = in_elipse1 <= 1
                    vol_img_ref[x_grid[cond], y_grid[cond], i_slice] = 255 / mul[cond]

        return vol_img_ref

    def get_vessel_end_loc(self, shape = None):
        """Generate the vessel end location"""
        if shape is None:
            shape = tuple()
        elif isinstance(shape, int):
            shape = (shape,)

        ray_cell_length = self.params.ray_cell_length
        ray_cell_variance = self.params.ray_cell_variance
        # ray_height = self.params.ray_height
        rcl_d3 = ray_cell_length / 3
        rcl_t2 = ray_cell_length * 2
        sie_x = self.params.size_im_enlarge[0]

        lim = sie_x + ray_cell_length

        vessel_end_loc = np.random.rand(*shape, 1) * ray_cell_length
        while np.any(vessel_end_loc[...,-1] < lim):
            tmp = ray_cell_length + ray_cell_variance * np.random.randn(*shape, 1)
            tmp[tmp < rcl_d3] = rcl_t2
            tmp[tmp > rcl_t2] = rcl_t2
            vessel_end_loc = np.concatenate((vessel_end_loc, vessel_end_loc[..., -1:] + tmp), axis=-1)
        vessel_end_loc = np.round(vessel_end_loc)

        return vessel_end_loc.astype(int)

    @Clock('ray_cell')
    def generate_raycell(
            self, ray_idx: int, ray_width: npt.NDArray, input_volume: npt.NDArray
        ) -> npt.NDArray:
        """Generate ray cell

        Args:
            ray_idx (int): y-index of the ray cell
            ray_width (npt.NDArray): Ray cell width
            input_volume (npt.NDArray): Input 3D gray-scale image volume to modify

        Returns:
            npt.NDArray: Modified 3D gray-scale image volume with ray cells
        """
        self.logger.info('=' * 80)
        self.logger.info('Generating ray cell...')
        vol_img_ref_final = np.copy(input_volume)

        ray_cell_length = self.params.ray_cell_length
        ray_height = self.params.ray_height

        rcl_d2 = ray_cell_length / 2
        rcl_d2_r = np.round(ray_cell_length / 2)

        sie_x, _, sie_z = self.params.size_im_enlarge
        # gx, gy = self.params.x_grid.shape

        x_grid_all = self.x_grid_all
        y_grid_all = self.y_grid_all
        thick_all = self.thickness_all

        cell_end_thick = self.params.cell_end_thick
        cet_d2_p1 = cell_end_thick // 2 + 1
        cet_d2_m1 = cell_end_thick // 2 - 1

        # ray_idx = np.array(ray_idx).flatten().astype(int)
        ray_width = np.array(ray_width).astype(int)

        dx = np.arange(sie_x)

        vessel_end_loc = self.get_vessel_end_loc(2)

        ray_column_rand = int(np.round(1 / 2 * ray_height))

        for m2, j_slice in enumerate(ray_width):
            # self.logger.debug('  %d/%d   %d', m2, len(ray_width), j_slice)
            k0 = j_slice
            k1 = j_slice + ray_column_rand
            tmp0_1 = (max(1, k0) + min(k0 + ray_height, sie_z)) / 2
            tmp1_1 = (max(1, k1) + min(k1 + ray_height, sie_z)) / 2

            t0 = int(np.round(tmp0_1)) - 1  # 0-indexed
            t1 = int(np.round(tmp1_1)) - 1

            for i, column_idx in enumerate([ray_idx, ray_idx + 1]):
                if column_idx % 2:
                    t, k, tmp_1 = t1, k1, tmp1_1
                else:
                    t, k, tmp_1 = t0, k0, tmp0_1

                if t < 0 or t >= sie_z - 11:
                    continue
                if t not in self.params.save_slice:
                    continue


                vel = vessel_end_loc[i]
                vel = vel[vel <= sie_x + rcl_d2]

                # rand = ray_column_rand * ((column_idx + 1) % 2)

                interp_x0 = x_grid_all[:, column_idx, t]
                interp_y0 = y_grid_all[:, column_idx, t]
                interp_x1 = x_grid_all[:, column_idx + 1, t]
                interp_y1 = y_grid_all[:, column_idx + 1, t]
                interp_thick = thick_all[:, column_idx, t]
                tmp_2 = rcl_d2_r if m2 % 2 else 0

                try:
                    y_interp1_c = CubicSpline(interp_x0, interp_y0)(dx) - 1.5
                    y_interp2_c = CubicSpline(interp_x1, interp_y1)(dx) + 1.5
                    thick_interp_c = CubicSpline(interp_x0, interp_thick)(dx)
                except ValueError:
                    # TODO: check how matlab handles this
                    # The spline interpolations as is now can generate an X/Y grid where 2 points can swap order
                    # e.g. on x-coords the left points can get pushed right enough and the right-point left enough
                    # for them to swap order causing successive spline interpolation to fail because the x-coords
                    # are not monotonic
                    self.logger.warning('    WARNING: Spline interpolation failed')
                    continue

                cell_center = np.column_stack((
                    dx,
                    np.round((y_interp2_c + y_interp1_c)) / 2,
                    np.full(dx.shape, tmp_1)
                ))
                cell_r = np.column_stack((
                    (y_interp2_c - y_interp1_c) / 2,
                    np.full(dx.shape, (np.min((k + ray_height, sie_z)) - np.max((1, k))) / 2)
                )) + 0.5

                d_col = int(cell_end_thick)
                for vel_r, vel_r1 in zip(vel[:-1], vel[1:]):
                    # tmp_2 = rcl_d2_r if m2 % 2 else 0
                    vel_col_r = int(vel_r + tmp_2)
                    if vel_col_r < 1:
                        continue
                    vel_col_r1 = int(vel_r1 + tmp_2)
                    if vel_col_r1 > sie_x - 1:
                        continue
                    cell_neigh_pt = np.array([
                        [vel_col_r, vel_col_r1],
                        [np.round(y_interp1_c[vel_col_r]), np.round(y_interp2_c[vel_col_r])],
                        [np.max((0, k)), np.min((k + ray_height, sie_x))]
                    ])

                    valid_idx = np.arange(
                        vel_col_r + d_col,
                        vel_col_r1 - cet_d2_p1  #Right inclusive
                    )
                    valid_idx = set(int(_) for _ in valid_idx)
                    d_col = cet_d2_m1

                    for idx in range(vel_col_r, vel_col_r1):
                        if j_slice == np.min(ray_width):
                            vol_img_ref_final[
                                idx,
                                int(y_interp1_c[idx]):int(y_interp2_c[idx] + 1),
                                int(cell_center[idx, 2]):int(cell_neigh_pt[2, 1] + 1)
                            ] = 255
                        elif j_slice == np.max(ray_width):
                            vol_img_ref_final[
                                idx,
                                int(y_interp1_c[idx]):int(y_interp2_c[idx] + 1),
                                int(cell_neigh_pt[2, 0]):int(cell_center[idx, 2])
                            ] = 255
                        else:
                            vol_img_ref_final[
                                idx,
                                int(y_interp1_c[idx]):int(y_interp2_c[idx] + 1),
                                int(cell_neigh_pt[2, 0]):int(cell_neigh_pt[2, 1])
                            ] = 255

                        if idx not in valid_idx:
                            continue

                        for j in range(int(cell_neigh_pt[1, 0]), int(cell_neigh_pt[1, 1]) + 1):
                            for s in range(int(cell_neigh_pt[2, 0]), int(cell_neigh_pt[2, 1]) + 1):
                                outer_elipse = (
                                    (j - cell_center[idx, 1])**2 / cell_r[idx, 0]**2 +
                                    (s - cell_center[idx, 2])**2 / cell_r[idx, 1]**2
                                )

                                if outer_elipse < 1:
                                    inner_elipse = (
                                        (j - cell_center[idx, 1])**2 / (cell_r[idx, 0] - thick_interp_c[idx])**2 +
                                        (s - cell_center[idx, 2])**2 / (cell_r[idx, 1] - thick_interp_c[idx])**2
                                    )
                                    # print('   ', j, s)
                                    vol_img_ref_final[idx, j, s] = int(
                                        (1 / (1 + np.exp(-(inner_elipse - 1) / .05))) * 255
                                    )

        return vol_img_ref_final

    @Clock('deformation')
    @Clock('deform:generate')
    def generate_deformation(self, ray_cell_idx: npt.NDArray, indx_skip_all: npt.NDArray, idx_vessel_cen: npt.NDArray):
        """Add complicated deformation to the volume image. The deformation fields are generated separately.
        Then, they are summed together. Here u, v are initialized to be zero. Then they are summed."""
        self.logger.info('=' * 80)
        self.logger.info('Generating deformation...')
        self.logger.debug('  ray_cell_idx: %s', ray_cell_idx.shape)
        sie_x, sie_y, _ = self.params.size_im_enlarge

        # lx, ly, _ = self.x_grid_all.shape
        gx, gy = self.params.x_grid.shape

        u = np.zeros((sie_x, sie_y), dtype=float)
        v = np.zeros_like(u, dtype=float)
        u1 = np.zeros_like(u, dtype=float)
        v1 = np.zeros_like(u, dtype=float)

        lx = (gx - 1) // 2
        ly = (gy - 1) // 2

        x_slice = self.x_grid_all[:, :, 0]
        y_slice = self.y_grid_all[:, :, 0]

        skip_idx = set()
        for ix, iy in indx_skip_all.reshape(-1, 2):
            if iy % 2 == 0:
                continue
            if (iy + 1) % 4 == 0:
                ix -= 1
            if ix % 2 == 0:
                continue
            skip_idx.add((x_slice[ix, iy], y_slice[ix, iy]))

        xc_grid = np.empty((lx, ly))
        yc_grid = np.empty_like(xc_grid)
        xc_grid[:, 0::2] = x_slice[1:-1:2, 1:-1:4]
        xc_grid[:, 1::2] = x_slice[2:  :2, 3:-1:4]
        yc_grid[:, 0::2] = y_slice[1:-1:2, 1:-1:4]
        yc_grid[:, 1::2] = y_slice[2:  :2, 3:-1:4]

        is_close_to_ray = np.zeros_like(xc_grid, dtype=bool)
        is_close_to_ray_far = np.zeros_like(xc_grid, dtype=bool)

        cond = np.zeros_like(xc_grid, dtype=bool)

        # idx_vessel_cen: (num_vessels, 2)
        for xc, yc in idx_vessel_cen:
            if yc % 2 == 0:
                continue
            if (yc + 1) % 4 == 0:
                xc -= 1
            if xc % 2 == 0:
                continue
            cond[xc // 2, yc // 2] = True

        if ray_cell_idx.size:
            for j in range(1, ly - 1, 2):
                mm = min(np.min(np.abs(j - ray_cell_idx)), np.min(np.abs(j - ray_cell_idx - 1)))
                is_close_to_ray[:, j // 2] = mm <= 4
                is_close_to_ray_far[:, j // 2] = mm <= 8

        s_grid = np.sign(np.random.randn(lx, ly))
        s_grid[cond] = -1
        k_grid = np.empty((lx, ly, 4))

        k_grid[..., 0] = 0.08
        k_grid[..., 1] = 0.06
        k_grid[..., 2] = 2
        k_grid[..., 3] = 15

        # k_grid[~cond, 0] = 0.08
        # k_grid[~cond, 1] = 0.06
        k_grid[~cond, 2] = 2 + np.random.rand(lx, ly)[~cond]
        k_grid[~cond, 3] = 1 + np.random.rand(lx, ly)[~cond]
        k_grid[~cond & is_close_to_ray, 3] *= 0.3

        k_grid[cond, 0] = 0.06
        k_grid[cond, 1] = 0.055
        # k_grid[cond, 2] = 2
        k_grid[cond & is_close_to_ray_far, 3] = 3 + 5 * np.random.rand(lx, ly)[cond & is_close_to_ray_far]
        # k_grid[cond & ~is_close_to_ray_far, 3] = 15

        for xc, yc, k, s in zip(xc_grid.flatten(), yc_grid.flatten(), k_grid.reshape(-1, 4), s_grid.flatten()):
            if (xc, yc) in skip_idx:
                continue
            xp, yp = dist.get_distortion_grid(xc, yc, sie_x, sie_y, self.local_distortion_cutoff)
            local_dist = dist.local_distort(xp, yp, xc, yc, k)
            u[xp, yp] += -s * local_dist

        k_grid[~cond, 2] = 2 + np.random.rand(lx, ly)[~cond]
        k_grid[~cond, 3] = 1 + np.random.rand(lx, ly)[~cond]
        k_grid[~cond & is_close_to_ray, 3] *= 0.3

        for xc, yc, k, s in zip(xc_grid.flatten(), yc_grid.flatten(), k_grid.reshape(-1, 4), s_grid.flatten()):
            if (xc, yc) in skip_idx:
                continue
            xp, yp = dist.get_distortion_grid(xc, yc, sie_x, sie_y, self.local_distortion_cutoff)
            local_dist = dist.local_distort(yp, xp, yc, xc, k)
            v[xp, yp] += -s * local_dist

        for xc, yc, cf in zip(xc_grid.flatten(), yc_grid.flatten(), is_close_to_ray_far.flatten()):
            if np.random.rand() >= 0.01:
                continue
            k = [0.01, 0.008, 1.5 * (1 + np.random.rand()), 0.2 * (1 + np.random.rand())]
            if not cf:
                k[3] *= 2.5

            xp, yp = dist.get_distortion_grid(xc, yc, sie_x, sie_y, self.local_distortion_cutoff)
            if np.random.randn() > 0:
                local_dist = dist.local_distort(xp, yp, xc, yc, k)
                u1[xp, yp] += np.sign(np.random.randn()) * local_dist
                # u1 += np.sign(np.random.randn()) * local_distort(x, y, xc, yc, k)
            else:
                local_dist = dist.local_distort(yp, xp, yc, xc, k)
                v1[xp, yp] += np.sign(np.random.randn()) * local_dist
                # v1 += np.sign(np.random.randn()) * local_distort(y, x, yc, xc, k)

        return u, v, u1, v1

    @Clock('deformation')
    @Clock('deform:rc_shrink')
    def ray_cell_shrinking(self, width: npt.NDArray, idx_all: npt.NDArray, dist_v: npt.NDArray) -> npt.NDArray:
        """Shrink the ray cell width"""
        self.logger.info('=' * 80)
        self.logger.info('Ray cell shrinking...')
        # grid_shape = self.params.x_grid.shape
        # y_vector = self.params.y_vector
        x_grid_all = self.x_grid_all
        y_grid_all = self.y_grid_all
        # thickness_all = self.thickness_all

        cell_thick = self.params.cell_wall_thick
        sie_x, sie_y, sie_z = self.params.size_im_enlarge

        ray_height = self.params.ray_height
        ray_size = self.params.cell_r - cell_thick / 2
        # slice_idx = self.params.save_slice
        # TODO: need to implement slice_idx with multiple indexes
        # I think this was reused from some other code because when the output array
        # is used it assumes it does not have a third dimension (or that it is always 1)

        cnt = defaultdict(int)
        for idx in idx_all.flatten():
            cnt[int(idx)] += 1

        dx = np.arange(sie_x, dtype=int)
        _, y_grid = np.mgrid[0:sie_x, 0:sie_y]

        v_all = np.zeros((sie_x, sie_y, len(self.params.save_slice)), dtype=float)
        for i, slice_idx in enumerate(self.params.save_slice):
            self.logger.debug('slice: %d/%d', i, len(self.params.save_slice))
            x_node_grid = x_grid_all[..., slice_idx]
            y_node_grid = y_grid_all[..., slice_idx]
            # self.logger.debug('x_node_grid.shape: %s', x_node_grid.shape)

            base_k = 0

            for key in cnt.keys():
                coeff1 = np.ones(sie_z)
                coeff2 = np.zeros(sie_z)
                v1 = np.zeros_like(y_grid, dtype=float)
                v2 = np.zeros_like(y_grid, dtype=float)

                v1_all = None
                v2_all = None

                self.logger.debug('   ray_idx = %d * (dupl = %d)', key, cnt[key])
                # This relies on the fact that idx_all is ordered by construction
                for key_cnt in range(cnt[key]):
                    k = base_k + key_cnt
                    idx = idx_all[k]

                    # self.logger.debug(f'  {idx=} {x_node_grid[:, idx].shape=} {y_node_grid[:, idx].shape=} {dx.shape=}')
                    y_node_grid_1 = CubicSpline(x_node_grid[:, idx], y_node_grid[:, idx])(dx)
                    y_node_grid_2 = CubicSpline(x_node_grid[:, idx + 2], y_node_grid[:, idx + 2])(dx)

                    v_node_grid_1 = dist_v[dx, np.round(y_node_grid_1).astype(int)]
                    v_node_grid_2 = dist_v[dx, np.round(y_node_grid_2).astype(int)]

                    R = (y_node_grid_2 + v_node_grid_2 - y_node_grid_1 - v_node_grid_1) / 2 - cell_thick / 2 + 1
                    y_center = (y_node_grid_2 + y_node_grid_1) / 2

                    for j in range(sie_x):
                        idx = np.arange(
                            max(cell_thick / 2, np.round(y_center[j] - R[j])),
                            min(y_center[j] + R[j], sie_y - cell_thick / 2 + 1) + 1,
                            dtype=int
                        )
                        if len(idx) > 2:
                            v1[j, idx] = -(idx - y_center[j])
                            v1[j, 0:idx[0]] = R[j]
                            v1[j, idx[-1]:] = -R[j]

                    min_idx_1 = np.max((1, np.round(np.min(width[k]) + ray_size))) - 1
                    max_idx_1 = np.min((np.round(np.max(width[k]) + ray_height - ray_size), sie_z)) - 1

                    idx_1 = []
                    if max_idx_1 - min_idx_1  > 10:
                        idx_1 = np.arange(min_idx_1, max_idx_1 + 1, dtype=int)

                    min_idx_3 = np.max((1, np.round(np.min(width[k]) - 3*ray_size))) - 1
                    max_idx_3 = np.min((np.round(np.min(width[k]) + 1*ray_size), sie_z)) - 1
                    idx_3 = []
                    if max_idx_3 - min_idx_3 > 10:
                        idx_3 = np.arange(min_idx_3, max_idx_3 + 1, dtype=int)

                    min_idx_4 = np.max((1, np.round(np.max(width[k]) + ray_height - ray_size))) - 1
                    max_idx_4 = np.min((np.round(np.max(width[k]) + ray_height + 3*ray_size), sie_z)) - 1
                    idx_4 = []
                    if max_idx_4 - min_idx_4 > 10:
                        idx_4 = np.arange(min_idx_4, max_idx_4 + 1, dtype=int)

                    coeff1[idx_1] = 0
                    coeff2[idx_1] = 1

                    if len(idx_3) > 5:
                        coeff1[idx_3] = np.arange(len(idx_3)-1, -1, -1) / np.round(4 * ray_size)
                        coeff2[idx_3] = np.arange(len(idx_3)) / np.round(4 * ray_size)
                    if len(idx_4) > 5:
                        coeff1[idx_4] = np.arange(len(idx_4)) / np.round(4 * ray_size)
                        coeff2[idx_4] = np.arange(len(idx_4)-1, -1, -1) / np.round(4 * ray_size)

                    for j in range(sie_x):
                        temp2 = 2 * R[j] * (1 / (1 + np.exp(0.02 * (y_grid[j,:] - y_center[j] + R[j]))) - 0.5)
                        v2[j, :int(y_center[j] - R[j])] = temp2[:int(y_center[j] - R[j])]
                        temp2 = 2 * R[j] * (1 / (1 + np.exp(0.02 * (y_grid[j,:] - y_center[j] - R[j]))) - 0.5)
                        v2[j, int(y_center[j] + R[j]):] = temp2[int(y_center[j] + R[j]):]

                    # Take either the last one with this condition or the first one at all
                    # The use of indicator[t] with [~,indx_sort] = sort(indicator, 'descend') will preserve the order
                    # according to the matlab documentation (meaning) that it is taking either the first indicator[t] == 1
                    # or the first indicator[t] == 0
                    if slice_idx < max_idx_4 and slice_idx > min_idx_1:
                        v1_all = v1
                        v2_all = v2
                        break
                    elif v1_all is None:
                        v1_all = v1
                        v2_all = v2

                base_k += cnt[key]
                v_all[:, :, i] += coeff1[slice_idx] * v1_all + coeff2[slice_idx] * v2_all

        return v_all

    @Clock('deformation')
    @Clock('deform:apply')
    def apply_deformation(self, slice_ref: npt.NDArray, u: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
        """Apply the deformation to the volume image"""
        self.logger.info('=' * 80)
        self.logger.info('Applying deformation...')
        sie_x, sie_y, _ = self.params.size_im_enlarge
        x_grid, y_grid = np.mgrid[0:sie_x, 0:sie_y]
        x_interp = x_grid + u
        y_interp = y_grid + v

        Vq = griddata(
            (x_interp.flatten(), y_interp.flatten()),
            slice_ref.flatten(),
            (x_grid, y_grid),
            method='linear'
        )

        img_interp = Vq.reshape(x_interp.shape)
        img_interp = np.clip(img_interp, 0, 255)

        return img_interp

    @Clock('Disk IO')
    def create_dirs(self):
        """Ensure the output directories are created"""
        for dir_name in ['volImgBackBone', 'LocalDistVolume', 'LocalDistVolumeDispU', 'LocalDistVolumeDispV']:
            os.makedirs(os.path.join(self.root_dir, dir_name), exist_ok=True)

    def save_slices(self, vol_img_ref: npt.NDArray, dirname: str):
        """Save the requested slice of the generated volume image"""
        self.logger.debug('vol_img_ref.shape: %s', vol_img_ref.shape)
        self.logger.debug('min/max: %f %f', np.min(vol_img_ref), np.max(vol_img_ref))
        for slice_idx in self.params.save_slice:
            filename = os.path.join(self.root_dir, dirname, f'volImgRef_{slice_idx+1:05d}.tiff')

            self.logger.debug('Saving slice %d to %s', slice_idx, filename)

            self.save_2d_img(vol_img_ref[:, :, slice_idx], filename)

    @staticmethod
    @Clock('Disk IO')
    def save_2d_img(data: npt.NDArray, filename: str):
        """Save 2D data to a TIFF file"""
        img = Image.fromarray(data.astype(np.uint8), mode='L')
        img.show()
        img.save(filename)

    @Clock('Disk IO')
    def save_distortion(self, u: npt.NDArray, v: npt.NDArray, slice_idx: int):
        """Save the distortion fields"""
        u_name = os.path.join(self.root_dir, 'LocalDistVolumeDispU', f'u_volImgRef_{slice_idx+1:05d}.csv')
        v_name = os.path.join(self.root_dir, 'LocalDistVolumeDispV', f'v_volImgRef_{slice_idx+1:05d}.csv')
        np.savetxt(u_name, np.round(u, decimals=4), delimiter=',')
        np.savetxt(v_name, np.round(v, decimals=4), delimiter=',')

    @abstractmethod
    def _generate(self):
        """Generate the volume image"""

    def generate(self):
        """Generate the volume image"""
        self.create_dirs()
        self._generate()

        Clock.report_all()
        self.logger.info('======== DONE ========')
