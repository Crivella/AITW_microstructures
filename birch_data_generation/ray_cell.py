"""Ray cells"""
import os
from collections import defaultdict

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.interpolate import CubicSpline, RegularGridInterpolator, griddata

from .clocks import Clock
from .fit_elipse import fit_elipse
from .loggers import logger
from .params import RayCellParams

dir_root = 'SaveBirch_'

def local_distort(
        x_grid: npt.NDArray, y_grid: npt.NDArray,
        x0: npt.NDArray, y0: npt.NDArray,
        k: tuple[float, float, float, float]
    ) -> npt.NDArray:
    """Local distortion of the image"""

    k1, k2, k3, k4 = k
    sigma = k3 / k2
    k0 = k4 / k2

    exp1 = np.exp(k1 * (x_grid - x0))
    exp2 = np.exp(k2 * (x_grid - x0))
    exp3 = np.exp(-((y_grid - y0) / sigma)**2 / 2)

    # u = exp1 / (1 + exp1) - exp2 / (1 + exp2)
    u = (exp1 - exp2) / (1 + exp1 + exp2 + exp1 * exp2)
    u *= k0 * exp3

    return u


class RayCell:
    local_distortion_cutoff = 200

    def __init__(self, params: RayCellParams):
        self.params = params

        self.x_grid_all = None
        self.y_grid_all = None
        self.thickness_all = None

        self._root_dir = None

    def get_ldist_grid(self, x_center: int, y_center: int):
        """Get the X,Y grids centered on (x_center, y_center) with cutoff of `local_distortion_cutoff`"""
        ldc = self.local_distortion_cutoff
        sie_x, sie_y, _ = self.params.size_im_enlarge
        x_grid, y_grid = np.mgrid[
            max(0, x_center - ldc):min(sie_x, x_center + ldc),
            max(0, y_center - ldc):min(sie_y, y_center + ldc)
        ].astype(int)

        return x_grid, y_grid

    def get_grid_all(self):
        """Specify the location of grid nodes and the thickness (with disturbance)"""
        gx, gy = self.params.x_grid.shape
        gz = self.params.size_im_enlarge[2]
        ds = self.params.slice_interest_space

        slice_interest = np.arange(0, gz, ds)
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

        self.x_grid_all = x_grid_all
        self.y_grid_all = y_grid_all
        self.thickness_all = thickness_all

        return x_grid_all, y_grid_all, thickness_all

    def get_ray_cell_indexes(self) -> npt.NDArray:
        """Get ray cell indexes"""
        ray_cell_x_ind_all = np.empty((1, 0))
        if self.params.is_exist_ray_cell:
            ray_cell_linspace = np.arange(9, len(self.params.y_vector) - 10, self.params.ray_space)
            ray_cell_x_ind_all = ray_cell_linspace + np.random.rand(len(ray_cell_linspace)) * 10 - 5
            ray_cell_x_ind_all = np.floor(ray_cell_x_ind_all / 2) * 2

        return ray_cell_x_ind_all.astype(int)

    def get_vessels_all(self, ray_cell_x_ind_all: npt.NDArray = None):
        """Get vessels"""
        # TODO: need testing
        logger.info('=' * 80)
        logger.info('Generating vessels...')
        if not self.params.is_exist_vessel:
            return np.empty((0, 2), dtype=int)
        x_vector = self.params.x_vector
        y_vector = self.params.y_vector

        x_rand_1 = np.round((np.random.rand(self.params.vessel_count) * (len(x_vector) - 16) + 8) / 2) * 2 - 1
        y_rand_1 = np.round((np.random.rand(self.params.vessel_count) * (len(y_vector) - 14) + 7) / 4) * 4
        y_rand_2 = np.round((np.random.rand(self.params.vessel_count // 2) * (len(y_vector) - 14) + 7) / 2) * 2
        x_rand_2 = np.round((np.random.rand(self.params.vessel_count // 2) * (len(x_vector) - 16) + 8) / 2) * 2

        x_rand_all = np.vstack((x_rand_1, x_rand_2))
        y_rand_all = np.vstack((y_rand_1, y_rand_2))
        vessel_all = np.column_stack((x_rand_all, y_rand_all))

        # Remove some vessel that too close to the other vessels
        vessel_all = self.vessel_filter_close(vessel_all)
        vessel_all = self.vessel_filter_ray_close(vessel_all, ray_cell_x_ind_all)
        vessel_all = self.vessel_extend(vessel_all)
        vessel_all = self.vessel_filter_ray_close(vessel_all, ray_cell_x_ind_all)

        return vessel_all.astype(int)

    def vessel_filter_close(self, vessel_all: npt.NDArray):
        """Filter the vessel that are too close to the other vessels"""
        all_idx = set()
        done = set()
        for i, vessel in enumerate(vessel_all):
            if vessel[0] == 0:
                continue
            dist = np.abs(vessel_all - vessel)
            mark0 = np.where((dist[:, 0] <= 6) & (dist[:, 1] <= 4))[0]
            if i not in done:
                all_idx.add(i)
            done.update(mark0)

        return vessel_all[list(all_idx)]

    def vessel_filter_ray_close(self, vessel_all: npt.NDArray, ray_cell_x_ind_all: npt.NDArray):
        """Filter the vessel that are too close to the ray cells"""
        all_idx = set()
        for i, vessel in enumerate(vessel_all):
            diff = vessel[0] - ray_cell_x_ind_all
            if not np.any((diff >= -3) & (diff <= 4)):
                all_idx.add(i)
        return vessel_all[list(all_idx)]

    def vessel_filter_ray_close2(self, vessel_all: npt.NDArray, ray_cell_x_ind_all: npt.NDArray):
        """Filter the vessel that too close to the ray cells"""
        lx = len(self.params.x_vector)
        ly = len(self.params.y_vector)

        all_idx = set()
        for i, vessel in enumerate(vessel_all):
            diff = vessel[0] - ray_cell_x_ind_all
            if not np.any((diff >= -3) & (diff <= 4)) and vessel[0] <= lx - 3 and vessel[1] <= ly - 3:
                all_idx.add(i)
        return vessel_all[list(all_idx)]

    def vessel_extend(self, vessel_all: npt.NDArray):
        """Extend the vessel"""
        vessel_all_extend = np.empty((0, 2))
        logger.info('-- Vessel extend --')
        logger.info('vessel_all: %s', vessel_all)
        for vessel in vessel_all:
            dist = vessel_all - vessel

            mark0 = np.where((dist[:, 0] <= 24) & (dist[:, 0] >= -8) & (np.abs(dist[:, 1]) <= 8))[0]
            mark1 = np.where((dist[:, 0] <= 12) & (dist[:, 0] >= -6) & (np.abs(dist[:, 1]) <= 6))[0]

            sign1 = np.random.choice([-1, 1])
            sign2 = np.random.choice([-1, 1])

            if len(mark0) > 1:
                vessel_all_extend = np.vstack((vessel_all_extend, vessel))
                possibility = np.random.rand(1)
                if len(mark1) <= 1:
                    if possibility < 0.2:
                        temp = [vessel[0] + 6 + sign1, vessel[1] + sign2 * 2]
                        vessel_all_extend = np.vstack((vessel_all_extend, temp))
                    else:
                        if possibility < 0.5:
                            temp = [vessel[0] + 6, vessel[1]]
                            vessel_all_extend = np.vstack((vessel_all_extend, temp))
            else:
                if vessel[0] + 12 < len(self.params.x_vector) and vessel[1] + 10 < len(self.params.y_vector):
                    temp0 = [vessel[0] + 5 + sign1, vessel[1]]
                    possibility = np.random.rand(1)
                    if possibility < 0.3:
                        temp = np.vstack((
                            temp0,
                            [temp0[0] + 5, temp0[1] + 2 * sign2]
                        ))
                    else:
                        temp = np.vstack((
                            temp0,
                            [temp0[0] + 5 + sign2, temp0[1]]
                        ))
                    vessel_all_extend = np.vstack((vessel_all_extend, vessel, temp))
                else:
                    vessel_all_extend = np.vstack((vessel_all_extend, vessel))
        return vessel_all_extend

    def fiber_filter_in_vessel(self, vessel_all: npt.NDArray):
        """This function is used to remove the fibers in the vessels"""
        # lx = len(self.params.x_vector)
        # ly = len(self.params.y_vector)
        num_vess = vessel_all.shape[0]  # (num_vess, 2)

        indx_skip_all = np.empty((num_vess, 6, 2), dtype=int)
        indx_skip_all[:, :, :] = vessel_all[:, np.newaxis, :]
        indx_skip_all += [
            (-1, -2),
            (+1, -2),
            (-2, +0),
            (+2, +0),
            (-1, +2),
            (+1, +2)
        ]

        indx_vessel = np.empty((num_vess, 6, 2))
        indx_vessel[:, :, :] = vessel_all[:, np.newaxis, :]
        indx_vessel += [
            (-3, -1),
            (-3, +1),
            (+0, -3),
            (+0, +3),
            (+3, -1),
            (+3, +1)
        ]

        # indx_skip_all = indx_skip_all[:,:,0] * ly + indx_skip_all[:,:,1]
        # indx_vessel = indx_vessel[:,:,0] * ly + indx_vessel[:,:,1]

        # indx_vessel_cen = vessel_all[:, 0] * ly + vessel_all[:, 1]

        indx_skip_all = indx_skip_all.reshape(-1, 2)
        indx_vessel = indx_vessel.reshape(-1, 2)

        return indx_skip_all.astype(int), indx_vessel.astype(int), vessel_all.astype(int)

    def distrbute_ray_cells(self, ray_cell_x_ind_all: npt.NDArray) -> tuple[
            npt.NDArray,
            list[npt.NDArray],
            npt.NDArray,
            npt.NDArray
        ]:
        """Distribute the ray cells across the volume

        Args:
            ray_cell_x_ind_all (npt.NDArray): Ray cell  indices

        Returns:
            tuple[ npt.NDArray, list[npt.NDArray], npt.NDArray, npt.NDArray ]:
            - Ray cell indices (num_ray_cells, 2): indices array A where A[j][1] = A[j][0] + 1
            - Ray cell widths (num_ray_cells, non_uniform): length of elements depends on the randomly generated group
            - Keep indices (num_ray_cells,): Index of idx in ray_cell_x_ind_all
            - Ray cell indices (num_ray_cells,): Array of indices of the ray cells (without the +1 column)
        """
        logger.info('=' * 80)
        logger.info('Distributing ray cells...')
        x_ind = []
        width = []
        # keep = []
        x_ind_all_update = []

        sie_z = self.params.size_im_enlarge[2]

        ray_cell_num = self.params.ray_cell_num
        ray_cell_num_std = self.params.ray_cell_num_std
        ray_height = self.params.ray_height

        m = int(np.ceil(sie_z / ray_cell_num / ray_height + 6))
        for i, idx in enumerate(ray_cell_x_ind_all):
            app = [0]
            ray_cell_space = np.round(16 * np.random.rand(m)) + 6
            rnd = np.round(-30 * np.random.rand())
            ray_idx = [idx, idx + 1]
            for rs in ray_cell_space:
                group = np.random.randn() * ray_cell_num_std + ray_cell_num
                group = np.clip(group, 5, 25)
                app = app[-1] + (np.arange(group + 1) + rs + rnd) * ray_height
                rnd = 0

                if app[0] > sie_z - 150:
                    break

                if app[-1] >= 150:
                    x_ind.append(ray_idx)
                    x_ind_all_update.append(idx)
                    width.append(np.round(app).astype(int))
                    # keep.append(i)

        return (
            np.array(x_ind, dtype=int),
            width,
            # np.array(width, dtype=float),
            # np.array(keep, dtype=int),
            np.array(x_ind_all_update, dtype=int)
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

    @Clock('small_fibers')
    def generate_small_fibers(
            self,
            skip_fiber_column: npt.NDArray,
            indx_skip_all: npt.NDArray,
            input_volume: npt.NDArray
        ) -> npt.NDArray:
        """Generate small fibers. This is a modified version of the original function.
        It runs in similar time but could be parallelized on the Z-slices and can compute only the
        required slices.

        Args:
            skip_fiber_column (npt.NDArray): indexes of columns where not to generate fibers
            indx_skip_all (npt.NDArray): indexes of grid where not to generate fibers
            input_volume (npt.NDArray): input 3D gray-scale image volume to modify

        Returns:
            npt.NDArray: modified 3D gray-scale image volume with small fibers
        """
        logger.info('=' * 80)
        logger.info('Generating small fibers...')
        vol_img_ref = np.copy(input_volume)

        neigh_loc = self.params.neighbor_local
        skip_fiber_column = np.array(skip_fiber_column).flatten().astype(int)
        skip_fiber_column = np.unique((skip_fiber_column[(skip_fiber_column % 2) == 1]) // 2)

        sie_x, sie_y, _ = self.params.size_im_enlarge
        gx, gy = self.params.x_grid.shape

        x_grid_all = self.x_grid_all
        y_grid_all = self.y_grid_all
        thick_all = self.thickness_all

        lx = (gx - 2) // 2
        ly = (gy - 2) // 2

        skip_idx = []
        for ix, iy in indx_skip_all.reshape(-1, 2):
            if iy % 2 == 0:
                continue
            if (iy + 1) % 4 == 0:
                ix += 1
            if ix % 2 == 0:
                continue
            skip_idx.append((ix // 2, iy // 2))
        skip_idx = np.array(skip_idx)

        point_coords = np.empty((lx, ly, 4, 2))
        t_all = np.empty((lx, ly))
        skip_cell_thick = 0  # TODO: Should this be a settable parameter?
        # for i_slice in range(sie_z):
        for i_slice in self.params.save_slice:
            logger.debug('  Small fibers: %d/%s', i_slice, self.params.save_slice)
            x_slice = x_grid_all[:,:, i_slice]
            y_slice = y_grid_all[:,:, i_slice]
            t_slice = thick_all[:,:, i_slice]

            # Assignments are split into [:, 0::2] and [1::2, :] to keep into account staggering along x direction
            # every other row
            t_all[:, 0::2] = t_slice[1:-2:2, 1:-2:4]
            t_all[:, 1::2] = t_slice[2:-1:2, 3:-2:4]

            for i,(dx,dy) in enumerate(neigh_loc.T):
                slice_1_1 = slice(1+dx, -2+dx, 2)
                slice_2_1 = slice(2+dx, (-1+dx) or None, 2)
                slice_1_2 = slice(1+dy, -2+dy, 4)
                slice_2_2 = slice(3+dy, -2+dy, 4)
                for j,s in enumerate((x_slice, y_slice)):
                    point_coords[:, 0::2, i,j] = s[slice_1_1, slice_1_2]
                    point_coords[:, 1::2, i,j] = s[slice_2_1, slice_2_2]

            if skip_cell_thick == 0:
                point_coords[:,:, 1, 1] -= 2
                point_coords[:,:, 3, 1] += 2

            r1, r2, h, k = fit_elipse(point_coords)  # Estimate the coefficients of the ellipse. (lx, ly, 4)

            # Set a very high value for h in nodes that should be ignored
            h[:, skip_fiber_column] = 80000
            if len(skip_idx) > 0:
                h[skip_idx[:, 0], skip_idx[:, 1]] = 80000
            h[self.get_fiber_end_condition(lx, ly, i_slice)] = 80000

            # The alternative is to write the full x/y grid and denote it into sub-domains based on the closest h/k
            # center and than use griddata to get the value of r1/r2/h/k on the full grid but this is slower
            for thick, _r1, _r2, _h, _k in zip(t_all.flatten(), r1.flatten(), r2.flatten(), h.flatten(), k.flatten()):
                if np.any(np.isnan([_h, _k, _r1, _r2])):
                    continue
                mr = np.floor(max(_r1, _r2))

                min_x = max(0, int(np.ceil(_h)) - mr)
                max_x = min(sie_x, int(np.ceil(_h)) + mr)
                min_y = max(0, int(np.ceil(_k)) - mr)
                max_y = min(sie_y, int(np.ceil(_k)) + mr)
                if min_x >= max_x or min_y >= max_y:
                    continue
                rx_grid, ry_grid = np.mgrid[min_x:max_x, min_y:max_y].astype(int)
                in_elipse_2 = (
                    (rx_grid - _h)**2 / (_r1 - thick - skip_cell_thick)**2 +
                    (ry_grid - _k)**2 / (_r2 - thick - skip_cell_thick)**2
                )

                vol_img_ref[rx_grid, ry_grid, i_slice] /= 1 + np.exp(-(in_elipse_2 - 1) * 20)

        return vol_img_ref.astype(int)

    @Clock('large_fibers')
    def generate_large_fibers(
            self,
            indx_vessel: npt.NDArray,
            indx_vessel_cen: npt.NDArray,
            indx_skip_all: npt.NDArray,
            input_volume: npt.NDArray
        ) -> npt.NDArray:
        """Generate large fibers."""
        logger.info('=' * 80)
        logger.info('Generating large fibers...')
        vol_img_ref = np.copy(input_volume)

        x_vector = self.params.x_vector
        y_vector = self.params.y_vector

        # TODO: !!! ensure index_vessel_* use the (n_idx, 2) shapes

        ly = len(y_vector)

        sie_x, sie_y, sie_z = self.params.size_im_enlarge

        vessel_length = self.params.vessel_length
        vessel_thicker = 1  # TODO: Should this be a settable parameter?
        vessel_length_variance = self.params.vessel_length_variance
        cell_end_thick = self.params.cell_end_thick

        # TODO: check and also work with 3D grid
        x_grid_all = self.x_grid_all.reshape(-1, sie_z)
        y_grid_all = self.y_grid_all.reshape(-1, sie_z)

        for i in range(1, len(x_vector)-2, 2):
            for j in range(1, len(y_vector)-2, 2):
                # The arrangement of the cells should be staggered. So every four nodes,
                # they should deviate along x direction.
                i1 = i + ((j + 1) % 4 == 0)
                idx = i1 * ly + j

                vessel_end_loc_all = [np.round(np.random.rand() * vessel_length)]
                for _ in range(int(np.ceil(sie_z / vessel_length)) + 8):
                    # temp = np.min(
                    #     3 * vessel_length,
                    #     np.max(100, vessel_length + np.random.randn() * self.params.vessel_length_variance)
                    # )
                    temp = np.clip(vessel_length + np.random.randn() * vessel_length_variance, 100, 3 * vessel_length)
                    vessel_end_loc_all.append(np.round(vessel_end_loc_all[-1] + temp))
                vessel_end_loc_all = np.array(vessel_end_loc_all)

                # This is a manually given value. To increase the randomness
                vessel_end_loc = vessel_end_loc_all - 4 * vessel_length + 1
                # The end of the vessel should be inside the volume.
                vessel_end_loc = vessel_end_loc[(vessel_end_loc >= 4) & (vessel_end_loc <= sie_z - 4)]

                vessel_end = np.empty((0, vessel_end_loc.size))
                for i in range(cell_end_thick):
                    vessel_end = np.vstack((vessel_end, vessel_end_loc + i))


                for i_slice in range(sie_z):
                    i_vessel = np.where(idx == indx_vessel_cen)[0]
                    if not i_vessel.size:
                        continue

                    # print('Large fibers:', i, j, i_slice)

                    point_coord = np.column_stack((
                        x_grid_all[*(*indx_vessel[i_vessel].T, i_slice)],
                        y_grid_all[*(*indx_vessel[i_vessel].T, i_slice)]
                    ))
                    r1, r2, h, k = fit_elipse(point_coord)  # Estimate the coeffecients of the elipse.

                    thick = self.thickness_all[idx, i_slice] + vessel_thicker
                    mr = np.floor(max(r1, r2))
                    for t1 in np.round(h) + np.arange(-mr, mr + 1):
                        for t2 in np.round(k) + np.arange(-mr, mr + 1):
                            if (t1 < 0 or t1 >= sie_x) or (t2 < 0 or t2 >= sie_y):
                                continue


                            # in_elipse0 = (t1 - h)**2 / r1**2 + (t2 - k)**2 / r2**2
                            in_elipse1 = (
                                (t1 - h)**2 / (r1 - thick * 4/3)**2 +
                                (t2 - k)**2 / (r2 - thick * 1)**2
                            )
                            in_elipse2 = (
                                (t1 - h)**2 / (r1 - thick * 5/3)**2 +
                                (t2 - k)**2 / (r2 - thick * 5/3)**2
                            )

                            if in_elipse1 <= 1:
                                # print('  ', t1, t2)
                                vol_img_ref[t1, t2, i_slice] = 1 / (1 + np.exp(-(in_elipse2 - 1) / 0.05)) * 255
        return vol_img_ref

    def get_vessel_end_loc(self, shape = None):

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
            # print('vessel_end_loc:', vessel_end_loc.shape, np.min(vessel_end_loc[..., -1]))
        vessel_end_loc = np.round(vessel_end_loc)
        # vessel_end_loc[vessel_end_loc > sie_x + ray_cell_length / 2] = np.nan

        # vessel_end_loc = vessel_end_loc[vessel_end_loc <= sie_x + ray_cell_length / 2]

        return vessel_end_loc.astype(int)

    @Clock('ray_cell')
    def generate_raycell(
            self, ray_idx: tuple[int, int], ray_width: npt.NDArray, input_volume: npt.NDArray
        ) -> npt.NDArray:
        """Generate ray cell

        Args:
            ray_idx (tuple[int, int]): Tuple of (idx, idx+1) where idx is the y-index of the ray cell
            ray_width (npt.NDArray): Ray cell width
            input_volume (npt.NDArray): Input 3D gray-scale image volume to modify

        Returns:
            npt.NDArray: Modified 3D gray-scale image volume with ray cells
        """
        logger.info('=' * 80)
        logger.info('Generating ray cell...')
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

        ray_idx = np.array(ray_idx).flatten().astype(int)
        ray_width = np.array(ray_width).flatten().astype(int)

        dx = np.arange(sie_x)

        vessel_end_loc = self.get_vessel_end_loc(len(ray_idx))

        ray_column_rand = int(np.round(1 / 2 * ray_height))

        for m2, j_slice in enumerate(ray_width):
            logger.debug('  %d/%d   %d', m2, len(ray_width), j_slice)
            k0 = j_slice
            k1 = j_slice + ray_column_rand
            tmp0_1 = (max(1, k0) + min(k0 + ray_height, sie_z)) / 2
            tmp1_1 = (max(1, k1) + min(k1 + ray_height, sie_z)) / 2
            # tmp_1 = (np.max((1, k)) + np.min((k + ray_height, sie_z))) / 2

            t0 = int(np.round(tmp0_1)) - 1  # 0-indexed
            t1 = int(np.round(tmp1_1)) - 1

            for i, column_idx in enumerate(ray_idx):
                # logger.debug('Ray cell: %d %s', column_idx, ray_idx)
                # vessel_end_loc = self.get_vessel_end_loc().reshape(-1)

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
                    logger.warning('    WARNING: Spline interpolation failed')
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

    def generate_deformation(self, ray_cell_idx: npt.NDArray, indx_skip_all: npt.NDArray, idx_vessel_cen: npt.NDArray):
        """Add complicated deformation to the volume image. The deformation fields are generated separately.
        Then, they are summed together. Here u, v are initialized to be zero. Then they are summed."""
        logger.info('=' * 80)
        logger.info('Generating deformation...')
        sie_x, sie_y, _ = self.params.size_im_enlarge

        lx, ly, _ = self.x_grid_all.shape
        gx, gy = self.params.x_grid.shape

        ldc = self.local_distortion_cutoff

        u = np.zeros((sie_x, sie_y), dtype=float)
        v = np.zeros_like(u, dtype=float)
        u1 = np.zeros_like(u, dtype=float)
        v1 = np.zeros_like(u, dtype=float)

        lx = (gx - 1) // 2
        ly = (gy - 1) // 2

        x_slice = self.x_grid_all[:, :, 0]
        y_slice = self.y_grid_all[:, :, 0]

        # TODO: check if this condition can be enforced geometrically
        skip_idx = set()
        for ix, iy in indx_skip_all.reshape(-1, 2):
            if iy % 2 == 0:
                continue
            if (iy + 1) % 4 == 0:
                ix += 1
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
                xc += 1
            if xc % 2 == 0:
                continue
            cond[xc // 2, yc // 2] = True

        for j in range(1, ly - 1, 2):
            mm = np.min(abs(j - ray_cell_idx))
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
            xp, yp = self.get_ldist_grid(xc, yc)
            local_dist = local_distort(xp, yp, xc, yc, k)
            u[xp, yp] += -s * local_dist

        logger.info('u shape: %s', u.shape)

        k_grid[~cond, 2] = 2 + np.random.rand(lx, ly)[~cond]
        k_grid[~cond, 3] = 1 + np.random.rand(lx, ly)[~cond]
        k_grid[~cond & is_close_to_ray, 3] *= 0.3

        for xc, yc, k, s in zip(xc_grid.flatten(), yc_grid.flatten(), k_grid.reshape(-1, 4), s_grid.flatten()):
            if (xc, yc) in skip_idx:
                continue
            xp, yp = self.get_ldist_grid(xc, yc)
            local_dist = local_distort(yp, xp, yc, xc, k)
            v[xp, yp] += -s * local_dist

        logger.info('v shape: %s', v.shape)

        for xc, yc, cf in zip(xc_grid.flatten(), yc_grid.flatten(), is_close_to_ray_far.flatten()):
            if np.random.rand() >= 0.01:
                continue
            k = [0.01, 0.008, 1.5 * (1 + np.random.rand()), 0.2 * (1 + np.random.rand())]
            if not cf:
                k[3] *= 2.5

            xp, yp = self.get_ldist_grid(xc, yc)
            if np.random.randn() > 0:
                local_dist = local_distort(xp, yp, xc, yc, k)
                u1[xp, yp] += np.sign(np.random.randn()) * local_dist
                # u1 += np.sign(np.random.randn()) * local_distort(x, y, xc, yc, k)
            else:
                local_dist = local_distort(yp, xp, yc, xc, k)
                v1[xp, yp] += np.sign(np.random.randn()) * local_dist
                # v1 += np.sign(np.random.randn()) * local_distort(y, x, yc, xc, k)

        return u, v, u1, v1

    def ray_cell_shrinking(self, width: npt.NDArray, idx_all: npt.NDArray, dist_v: npt.NDArray) -> npt.NDArray:
        """Shrink the ray cell width"""
        logger.info('=' * 80)
        logger.info('Ray cell shrinking...')
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
            x_node_grid = x_grid_all[..., slice_idx]
            y_node_grid = y_grid_all[..., slice_idx]
            logger.debug('x_node_grid.shape: %s', x_node_grid.shape)

            base_k = 0

            for key in cnt.keys():
                coeff1 = np.ones(sie_z)
                coeff2 = np.zeros(sie_z)
                v1 = np.zeros_like(y_grid, dtype=float)
                v2 = np.zeros_like(y_grid, dtype=float)

                v1_all = None
                v2_all = None

                logger.debug('  Ray cell shrinking: key = %d  cnt[key] = %d', key, cnt[key])
                # This relies on the fact that idx_all is ordered by construction
                for key_cnt in range(cnt[key]):
                    k = base_k + key_cnt
                    idx = idx_all[k]

                    logger.debug(f'  {idx=} {x_node_grid[:, idx].shape=} {y_node_grid[:, idx].shape=} {dx.shape=}')
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

    def apply_deformation(self, slice_ref: npt.NDArray, u: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
        """Apply the deformation to the volume image"""
        logger.info('=' * 80)
        logger.info('Applying deformation...')
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

    @property
    def root_dir(self):
        if self._root_dir is None:
            dir_cnt = 0
            while os.path.exists(f'{dir_root}{dir_cnt}'):
                dir_cnt += 1
            self._root_dir = f'{dir_root}{dir_cnt}'
        return self._root_dir

    def create_dirs(self):
        """Ensure the output directories are created"""
        for dir_name in ['volImgBackBone', 'LocalDistVolume', 'LocalDistVolumeDispU', 'LocalDistVolumeDispV']:
            os.makedirs(os.path.join(self.root_dir, dir_name), exist_ok=True)

    def save_slice(self, vol_img_ref: npt.NDArray, dirname: str):
        """Save the requested slice of the generated volume image"""
        logger.debug('vol_img_ref.shape: %s', vol_img_ref.shape)
        logger.debug('min/max: %f %f', np.min(vol_img_ref), np.max(vol_img_ref))
        for slice_idx in self.params.save_slice:
            filename = os.path.join(self.root_dir, dirname, f'volImgRef_{slice_idx+1:05d}.tiff')

            logger.debug('Saving slice %d to %s', slice_idx, filename)

            self.save_2d_img(vol_img_ref[:, :, slice_idx], filename)

    @staticmethod
    def save_2d_img(data: npt.NDArray, filename: str):
        """Save 2D data to a TIFF file"""
        img = Image.fromarray(data.astype(np.uint8), mode='L')
        # img.show()
        img.save(filename)

    def save_distortion(self, u: npt.NDArray, v: npt.NDArray, slice_idx: int):
        """Save the distortion fields"""
        u_name = os.path.join(self.root_dir, 'LocalDistVolumeDispU', f'u_volImgRef_{slice_idx+1:05d}.csv')
        v_name = os.path.join(self.root_dir, 'LocalDistVolumeDispV', f'v_volImgRef_{slice_idx+1:05d}.csv')
        np.savetxt(u_name, np.round(u, decimals=4), delimiter=',')
        np.savetxt(v_name, np.round(v, decimals=4), delimiter=',')

    def generate(self):
        """Generate ray cells"""
        np.random.seed(self.params.random_seed)

        self.get_grid_all()

        logger.info('PARAM: size_im_enlarge: %s', self.params.size_im_enlarge)
        logger.info('PARAM: x_vector.shape: %s', self.params.x_vector.shape)
        logger.info('PARAM: y_vector.shape: %s', self.params.y_vector.shape)
        logger.info('PARAM: x_grid_all.shape: %s', self.x_grid_all.shape)

        ray_cell_x_ind_all = self.get_ray_cell_indexes()
        logger.debug('ray_cell_x_ind_all.shape: %s', ray_cell_x_ind_all.shape)
        logger.debug('ray_cell_x_ind_all: %s', ray_cell_x_ind_all)

        vessel_all = self.get_vessels_all(ray_cell_x_ind_all)
        logger.debug('vessel_all.shape: %s', vessel_all.shape)
        logger.debug('vessel_all: %s', vessel_all)

        indx_skip_all, indx_vessel, indx_vessel_cen = self.fiber_filter_in_vessel(vessel_all)
        logger.debug('indx_skip_all: %s', indx_skip_all.shape)
        logger.debug('indx_vessel: %s', indx_vessel.shape)
        logger.debug('indx_vessel_cen: %s', indx_vessel_cen.shape)

        ray_cell_x_ind, ray_cell_width, ray_cell_x_ind_all_update = self.distrbute_ray_cells(ray_cell_x_ind_all)
        logger.debug('ray_cell_x_ind: %s  %s', ray_cell_x_ind.shape, ray_cell_x_ind)
        logger.debug('ray_cell_width:')

        for i,width in enumerate(ray_cell_width):
            logger.debug('   %d %s', i+1, width)
        logger.debug('ray_cell_x_ind_all_update: %s  %s', ray_cell_x_ind_all_update.shape, ray_cell_x_ind_all_update)

        vol_img_ref = np.full(self.params.size_im_enlarge, 255, dtype=int)
        vol_img_ref = self.generate_small_fibers(ray_cell_x_ind, indx_skip_all, vol_img_ref)
        vol_img_ref = self.generate_large_fibers(indx_vessel, indx_vessel_cen, indx_skip_all, vol_img_ref)

        if self.params.is_exist_ray_cell:
            for idx, width in zip(ray_cell_x_ind, ray_cell_width):
                logger.debug('Generating ray cell: %s / %s', idx, width)
                vol_img_ref = self.generate_raycell(idx, width, vol_img_ref)

        # Save the generated volume
        self.create_dirs()
        self.save_slice(vol_img_ref, 'volImgBackBone')

        # u1 and v1 are in a commented part of the code. Prob used in original code?
        u, v, _, _ = self.generate_deformation(ray_cell_x_ind, indx_skip_all, indx_vessel_cen)
        logger.debug('u.shape: %s  min/max: %s %s', u.shape, u.min(), u.max())
        logger.debug('v.shape: %s  min/max: %s %s', v.shape, v.min(), v.max())

        if self.params.is_exist_ray_cell:
            v_all_ray = self.ray_cell_shrinking(ray_cell_width, ray_cell_x_ind_all_update, v)
            v = v[..., np.newaxis] + v_all_ray
            logger.debug('vray   : %s  min/max: %s %s', v_all_ray.shape, v_all_ray.min(), v_all_ray.max())
            logger.debug('v.shape: %s  min/max: %s %s', v.shape, v.min(), v.max())

        for i, slice_idx in enumerate(self.params.save_slice):
            logger.debug('Saving deformation for slice %d', slice_idx)
            if self.params.is_exist_ray_cell:
                v_slice = v[..., i]
            else:
                v_slice = v
            self.save_distortion(u, v_slice, slice_idx)

            img_interp = self.apply_deformation(vol_img_ref[..., slice_idx], u, v_slice)

            filename = os.path.join(self.root_dir, 'LocalDistVolume', f'volImgRef_{slice_idx+1:05d}.tiff')
            self.save_2d_img(img_interp, filename)

        logger.info('======== DONE ========')
