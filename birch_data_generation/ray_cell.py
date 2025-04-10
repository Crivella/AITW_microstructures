"""Ray cells"""
import os
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.interpolate import (CubicSpline, NearestNDInterpolator,
                               RegularGridInterpolator, griddata)

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

    u = exp1 / (1 + exp1) - exp2 / (1 + exp2)
    u *= k0 * exp3

    return u

# TODO: enforce same same convention of using 2D grid or flattened grid everywhere

class RayCell:
    def __init__(self, params: RayCellParams):
        self.params = params

        self.x_grid_all = None
        self.y_grid_all = None
        self.thickness_all = None

        self._root_dir = None

    def get_grid_all(self):
        """Specify the location of grid nodes and the thickness (with disturbance)"""
        slice_interest = np.arange(0, self.params.size_im_enlarge[2], self.params.slice_interest_space)

        l = len(slice_interest)

        # grid_shape = self.params.x_grid.shape
        grid_size = self.params.x_grid.size

        x_grid = self.params.x_grid.flatten()
        y_grid = self.params.y_grid.flatten()

        x_grid_interp = np.random.rand(grid_size, l) * 3 - 1.5 + x_grid[:, np.newaxis]
        y_grid_interp = np.random.rand(grid_size, l) * 3 - 1.5 + y_grid[:, np.newaxis]
        thickness_interp = np.random.rand(grid_size, l) + self.params.cell_wall_thick - 0.5

        interp_x = np.arange(1, self.params.size_im_enlarge[2] + 1)
        x_grid_all = np.empty((grid_size, len(interp_x)))
        y_grid_all = np.empty((grid_size, len(interp_x)))
        thickness_all = np.empty((grid_size, len(interp_x)))
        for i in range(x_grid.size):
            x_grid_all[i, :] = CubicSpline(slice_interest, x_grid_interp[i, :])(interp_x)
            y_grid_all[i, :] = CubicSpline(slice_interest, y_grid_interp[i, :])(interp_x)
            thickness_all[i, :] = CubicSpline(slice_interest, thickness_interp[i, :])(interp_x)

        self.x_grid_all = x_grid_all
        self.y_grid_all = y_grid_all
        self.thickness_all = thickness_all

        return x_grid_all, y_grid_all, thickness_all

    # Test with RegularGridInterpolator
    def get_grid_all_(self):
        """Specify the location of grid nodes and the thickness (with disturbance)"""
        slice_interest = np.arange(0, self.params.size_im_enlarge[2], self.params.slice_interest_space)

        l = len(slice_interest)

        sie_z = self.params.size_im_enlarge[2]

        x_vector = self.params.x_vector
        y_vector = self.params.y_vector

        gx, gy = self.params.x_grid.shape

        shape_3d = (gx, gy, l)

        x_grid = self.params.x_grid#.flatten()
        y_grid = self.params.y_grid#.flatten()

        x_grid_interp = np.random.rand(*shape_3d) * 3 - 1.5 + x_grid[:,:,np.newaxis]
        y_grid_interp = np.random.rand(*shape_3d) * 3 - 1.5 + y_grid[:,:,np.newaxis]
        thickness_interp = np.random.rand(*shape_3d) + self.params.cell_wall_thick - 0.5

        interp_z = np.arange(sie_z)
        x, y, z = x_vector, y_vector, slice_interest
        Xi, Yi, Zi = np.meshgrid(x_vector, y_vector, interp_z, indexing='ij')

        logger.debug(f'Xi: {Xi.shape}, Yi: {Yi.shape}, Zi: {Zi.shape}')
        logger.debug(f'x_grid_interp: {x_grid_interp.shape}')

        interp = RegularGridInterpolator(
            (x, y, z), x_grid_interp.reshape(shape_3d),
            bounds_error=False, method='cubic'
        )
        x_grid_all = interp(np.stack((Xi, Yi, Zi), axis=-1))
        interp = RegularGridInterpolator(
            (x, y, z), y_grid_interp.reshape(shape_3d),
            bounds_error=False, method='cubic'
        )
        y_grid_all = interp(np.stack((Xi, Yi, Zi), axis=-1))
        interp = RegularGridInterpolator(
            (x, y, z), thickness_interp.reshape(shape_3d),
            bounds_error=False, method='cubic'
        )
        thickness_all = interp(np.stack((Xi, Yi, Zi), axis=-1))

        logger.debug(f'x_grid_all: {x_grid_all.shape}')
        logger.debug(f'y_grid_all: {y_grid_all.shape}')
        logger.debug(f'thickness_all: {thickness_all.shape}')
        logger.debug(f'x_grid_all[:,44,359]: {x_grid_all[:,44,359]}')


        self.x_grid_all = x_grid_all
        self.y_grid_all = y_grid_all
        self.thickness_all = thickness_all

        return x_grid_all, y_grid_all, thickness_all

    def get_ray_cell_indexes(self) -> npt.NDArray:
        """Get ray cell indexes"""
        ray_cell_x_ind_all = np.empty((1, 0))
        if self.params.is_exist_ray_cell:
            # TODO: Check if the range here should be 9, -10
            # ray_cell_linspace = np.arange(10, len(self.params.y_vector) - 9, self.params.ray_space)
            ray_cell_linspace = np.arange(9, len(self.params.y_vector) - 10, self.params.ray_space)
            ray_cell_x_ind_all = ray_cell_linspace + np.random.rand(len(ray_cell_linspace)) * 10 - 5
            # ray_cell_x_ind_all = np.floor(ray_cell_x_ind_all / 2) * 2 + 1
            ray_cell_x_ind_all = np.floor(ray_cell_x_ind_all / 2) * 2
            # ray_cell_x_ind_all = ray_cell_x_ind_all // 2  # + 1 array not 1-indexed

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
        ly = len(self.params.y_vector)
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

        indx_skip_all = indx_skip_all[:,:,0] * ly + indx_skip_all[:,:,1]
        indx_vessel = indx_vessel[:,:,0] * ly + indx_vessel[:,:,1]

        indx_vessel_cen = vessel_all[:, 0] * ly + vessel_all[:, 1]

        return indx_skip_all.astype(int), indx_vessel.astype(int), indx_vessel_cen.astype(int)

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

        # `keep` is only used in generate_small_fibers to basically reconstruct the array `x_ind`
        # Is it actually needed?

    @Clock('small_fibers')
    def generate_small_fibers_(
            self,
            skip_fiber_column: npt.NDArray,
            indx_skip_all: npt.NDArray,
            input_volume: npt.NDArray
        ) -> npt.NDArray:
        """Generate small fibers.

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

        skip_fiber_column = set(int(_) for _ in skip_fiber_column.flatten())
        indx_skip_all = set(int(_) for _ in indx_skip_all.flatten())

        sie_x, sie_y, sie_z = self.params.size_im_enlarge

        gx, gy = self.params.x_grid.shape

        neigh_loc = self.params.neighbor_local

        cell_length = self.params.cell_length
        cell_length_variance = self.params.cell_length_variance
        cell_end_thick = self.params.cell_end_thick

        num_fiber_end_loc = int(np.ceil(sie_z / cell_length)) + 7

        for i in range(1, gx - 2, 2):
            logger.debug('  Small fibers: %d/%d', i, gx-2)
            for j in range(1, gy - 2, 2):
                if j in skip_fiber_column:
                    continue
                # The arrangement of the cells should be staggered. So every four nodes,
                # they should deviate along x direction.
                i1 = i + ((j+1) % 4 == 0)
                idx = i1 * gy + j
                if idx in indx_skip_all:
                    # if this node exist on the surface of vessels, skip it.
                    continue

                initial = np.round(np.random.rand() * cell_length)
                fiber_end_loc_all = [initial,]

                for _ in range(num_fiber_end_loc):
                    temp = np.clip(cell_length + np.random.randn() * cell_length_variance, 100, 3*cell_length)
                    fiber_end_loc_all.append(np.round(fiber_end_loc_all[-1] + temp))
                fiber_end_loc_all = np.array(fiber_end_loc_all)

                # This is a manually given value. To increase the randomness
                fiber_end_loc = fiber_end_loc_all - 4 * cell_length + 1
                # The end of the vessel should be inside the volume.
                fiber_end_loc = fiber_end_loc[(fiber_end_loc >= 4) & (fiber_end_loc <= sie_z - 4)]

                fiber_end = np.empty((0, fiber_end_loc.size))
                for k in range(cell_end_thick):
                    fiber_end = np.vstack((fiber_end, fiber_end_loc + k))
                fiber_end = set(int(_) for _ in fiber_end.flatten())

                # Skip some fibers
                skip_cell_thick = 0  # TODO: Should this be a settable parameter?
                for i_slice in range(sie_z):
                    if i_slice in fiber_end:
                        # if this slice is not the end of this cell inside the lumen, it should be black
                        continue
                    # used to store the four neighbored points for ellipse fitting
                    point_coord = np.empty((4, 2))

                    for k in range(4):
                        # neigh_idx = len(x_vector) * (j + neigh_loc[1, k]) + i1 + neigh_loc[0, k]
                        neigh_idx = idx + neigh_loc[0, k] * gy + neigh_loc[1, k]
                        point_coord[k, :] = [self.x_grid_all[neigh_idx, i_slice], self.y_grid_all[neigh_idx, i_slice]]

                    # make the elipse more elipse
                    if skip_cell_thick == 0:
                        point_coord[1, 1] -= 2
                        point_coord[3, 1] += 2

                    r1, r2, h, k = fit_elipse(point_coord)  # Estimate the coefficients of the ellipse.

                    # Then we can estimate the diameter along two direction for the fiber
                    # The rectangle region covering the ellipse.
                    mr = np.floor(max(r1, r2))
                    # TODO: check if it is normal to obtain NaNs here and if matlab is also just ignoring them
                    if np.isnan(mr):
                        continue
                    region_cell_ind_x = int(np.ceil(h)) + np.arange(-mr, mr + 1)
                    region_cell_ind_y = int(np.ceil(k)) + np.arange(-mr, mr + 1)
                    region_cell_ind_x = np.arange(
                        np.max((0, np.min(region_cell_ind_x))),
                        np.min((sie_x, np.max(region_cell_ind_x))),
                        dtype=int
                    )
                    region_cell_ind_y = np.arange(
                        np.max((0, np.min(region_cell_ind_y))),
                        np.min((sie_y, np.max(region_cell_ind_y))),
                        dtype=int
                    )

                    if len(region_cell_ind_x) == 0 or len(region_cell_ind_y) == 0:
                        continue

                    region_cell_x, region_cell_y = np.meshgrid(region_cell_ind_x, region_cell_ind_y, indexing='ij')
                    # Internal contour of the elipse
                    in_elipse2 = (
                        (region_cell_x - h)**2 / (r1 - self.thickness_all[idx, i_slice] - skip_cell_thick)**2 +
                        (region_cell_y - k)**2 / (r2 - self.thickness_all[idx, i_slice] - skip_cell_thick)**2
                    )

                    vol_img_ref[region_cell_x, region_cell_y, i_slice] /= 1 + np.exp(-(in_elipse2 - 1) / 0.05)

        vol_img_ref = np.astype(vol_img_ref, int)
        vol_img_ref = np.clip(vol_img_ref, 0, 255)

        return vol_img_ref

    @Clock('small_fibers')
    def generate_small_fibers(
            self,
            skip_fiber_column: npt.NDArray,
            indx_skip_all: npt.NDArray,
            input_volume: npt.NDArray
        ) -> npt.NDArray:
        """Generate small fibers.

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

        print('skip_fiber_column:', skip_fiber_column.shape)
        print(skip_fiber_column)
        skip_fiber_column = np.array(skip_fiber_column).flatten().astype(int)
        skip_fiber_column = np.unique((skip_fiber_column[(skip_fiber_column % 2) == 1]) // 2)
        print(skip_fiber_column)
        # skip_fiber_column = set(int(_) for _ in skip_fiber_column.flatten())
        indx_skip_all = indx_skip_all.flatten().astype(int)
        sie_x, sie_y, sie_z = self.params.size_im_enlarge

        gx, gy = self.params.x_grid.shape

        neigh_loc = self.params.neighbor_local

        cell_length = self.params.cell_length
        cell_length_variance = self.params.cell_length_variance
        cell_end_thick = self.params.cell_end_thick

        num_fiber_end_loc = int(np.ceil(sie_z / cell_length)) + 7

        x_grid_all = self.x_grid_all.reshape((gx, gy, sie_z))
        y_grid_all = self.y_grid_all.reshape((gx, gy, sie_z))

        lx = (gx - 2) // 2
        ly = (gy - 2) // 2

        x_all_0 = np.empty((lx, ly))
        y_all_0 = np.empty_like(x_all_0)
        x_all_1 = np.empty_like(x_all_0)
        y_all_1 = np.empty_like(x_all_0)
        x_all_2 = np.empty_like(x_all_0)
        y_all_2 = np.empty_like(x_all_0)
        x_all_3 = np.empty_like(x_all_0)
        y_all_3 = np.empty_like(x_all_0)
        x_all_4 = np.empty_like(x_all_0)
        y_all_4 = np.empty_like(x_all_0)

        skip_idx = []
        for idx in indx_skip_all:
            x = idx // gy
            y = idx % gy
            if x % 2 and y % 2:
                skip_idx.append((x // 2, y // 2))
        skip_idx = np.array(skip_idx)
        print('skip_idx:', skip_idx.shape)

        skip_cell_thick = 0  # TODO: Should this be a settable parameter?
        # for i_slice in range(sie_z):
        for i_slice in range(min(3, sie_z)):
            start = time.time()
            logger.debug('  Small fibers: %d/%d', i_slice, sie_z)
            x_slice = x_grid_all[:,:, i_slice]
            y_slice = y_grid_all[:,:, i_slice]
            t_slice = self.thickness_all[:, i_slice]

            # Assignments are split into [:, 0::2] and [1::2, :] to keep into account staggering along x direction
            # every other row
            # (0, 0)
            x_all_0[:, 0::2] = x_slice[1:-2:2, 1:-2:4]
            x_all_0[:, 1::2] = x_slice[2:-1:2, 3:-2:4]
            y_all_0[:, 0::2] = y_slice[1:-2:2, 1:-2:4]
            y_all_0[:, 1::2] = y_slice[2:-1:2, 3:-2:4]
            # Neighbor (-1, 0)
            x_all_1[:, 0::2] = x_slice[0:-3:2, 1:-2:4]
            x_all_1[:, 1::2] = x_slice[1:-2:2, 3:-2:4]
            y_all_1[:, 0::2] = y_slice[0:-3:2, 1:-2:4]
            y_all_1[:, 1::2] = y_slice[1:-2:2, 3:-2:4]
            # Neighbor (1, 0)
            x_all_2[:, 0::2] = x_slice[2:-1:2, 1:-2:4]
            x_all_2[:, 1::2] = x_slice[3:  :2, 3:-2:4]
            y_all_2[:, 0::2] = y_slice[2:-1:2, 1:-2:4]
            y_all_2[:, 1::2] = y_slice[3:  :2, 3:-2:4]
            # Neighbor (0, -1)
            x_all_3[:, 0::2] = x_slice[1:-2:2, 0:-3:4]
            x_all_3[:, 1::2] = x_slice[2:-1:2, 2:-3:4]
            y_all_3[:, 0::2] = y_slice[1:-2:2, 0:-3:4]
            y_all_3[:, 1::2] = y_slice[2:-1:2, 2:-3:4]
            # Neighbor (0, 1)
            x_all_4[:, 0::2] = x_slice[1:-2:2, 2:-1:4]
            x_all_4[:, 1::2] = x_slice[2:-1:2, 4:-1:4]
            y_all_4[:, 0::2] = y_slice[1:-2:2, 2:-1:4]
            y_all_4[:, 1::2] = y_slice[2:-1:2, 4:-1:4]

            # This is after to properly do ellipse fit with neighbors and skip only based on centers
            # x_slice[:, skip_fiber_column] = np.nan
            # if len(skip_idx) > 0:
            #     x_slice[skip_idx[:, 0], skip_idx[:, 1]] = np.nan

            nxt = time.time()
            print('COORDS:', nxt - start)
            start = nxt

            print(x_grid_all.shape, y_grid_all.shape)
            print(x_all_1.shape, y_all_1.shape)

            x = np.stack((x_all_1, x_all_2, x_all_3, x_all_4), axis=-1)
            y = np.stack((y_all_1, y_all_2, y_all_3, y_all_4), axis=-1)
            print(x.shape)
            point_coords = np.stack((x, y), axis=-1)
            if skip_cell_thick == 0:
                point_coords[..., 1, 1] -= 2
                point_coords[..., 3, 1] += 2

            print(point_coords.shape)
            print(np.concatenate((point_coords**2, point_coords), axis=-1).shape)

            r1, r2, h, k = fit_elipse(point_coords)  # Estimate the coefficients of the ellipse.
            print(r1.shape)

            # r1[:, skip_fiber_column] = np.nan
            # r2[:, skip_fiber_column] = np.nan
            # Just set a very high value for h in nodes that should be ignored
            h[:, skip_fiber_column] = 80000
            if len(skip_idx) > 0:
                h[skip_idx[:, 0], skip_idx[:, 1]] = 80000
            # k[:, skip_fiber_column] = 80000
            # r2[w] = np.nan
            # h[w] = np.nan
            # k[w] = np.nan

            nxt = time.time()
            print('ELLIPSE FIT:', nxt - start)
            start = nxt

            x_grid, y_grid = np.mgrid[0:sie_x, 0:sie_y]

            # r1_grid = griddata(
            #     (x_all_0.flatten(), y_all_0.flatten()), r1.flatten(),
            #     (x_grid, y_grid),
            #     method='nearest'
            # )
            r1_grid = NearestNDInterpolator(
                (x_all_0.flatten(), y_all_0.flatten()), r1.flatten()
            )(x_grid, y_grid)
            nxt = time.time()
            print('GRIDS1:', nxt - start)
            start = nxt
            r2_grid = griddata(
                (x_all_0.flatten(), y_all_0.flatten()), r2.flatten(),
                (x_grid, y_grid),
                method='nearest'
            )
            nxt = time.time()
            print('GRIDS2:', nxt - start)
            start = nxt
            h_grid = griddata(
                (x_all_0.flatten(), y_all_0.flatten()), h.flatten(),
                (x_grid, y_grid),
                method='nearest'
            )
            nxt = time.time()
            print('GRIDS3:', nxt - start)
            start = nxt
            k_grid = griddata(
                (x_all_0.flatten(), y_all_0.flatten()), k.flatten(),
                (x_grid, y_grid),
                method='nearest'
            )
            nxt = time.time()
            print('GRIDS4:', nxt - start)
            start = nxt
            t_grid = griddata(
                (self.params.x_grid.flatten(), self.params.y_grid.flatten()), t_slice.flatten(),
                (x_grid, y_grid),
                method='nearest'
            )
            nxt = time.time()
            print('GRIDS5:', nxt - start)
            start = nxt

            in_ellipse_2 = (
                (x_grid - h_grid)**2 / (r1_grid - t_grid - skip_cell_thick)**2 +
                (y_grid - k_grid)**2 / (r2_grid - t_grid - skip_cell_thick)**2
            )
            v_grid = 1 + np.exp(-(in_ellipse_2 - 1) * 20)


            # TODO: implement skipping of (x,y) specific indexes
            # Prob can just set the values of r1, r2, h, k to NaN or 0

            # w = np.where(~np.isnan(v_grid))
            # vol_img_ref[*w, i_slice] /= v_grid[*w]
            vol_img_ref[..., i_slice] /= v_grid

            print('UPDATE:', time.time() - start)
            # break

        return np.clip(vol_img_ref, 0, 255).astype(int)

    def generate_large_fibers(
            self,
            indx_vessel: npt.NDArray,
            indx_vessel_cen: npt.NDArray,
            # x_ind: npt.NDArray,
            # keep: npt.NDArray,
            # ray_idx: npt.NDArray,
            indx_skip_all: npt.NDArray,
            input_volume: npt.NDArray
        ) -> npt.NDArray:
        """Generate large fibers."""
        logger.info('=' * 80)
        logger.info('Generating large fibers...')
        vol_img_ref = np.copy(input_volume)

        x_vector = self.params.x_vector
        y_vector = self.params.y_vector

        ly = len(y_vector)

        sie_x, sie_y, sie_z = self.params.size_im_enlarge

        vessel_length = self.params.vessel_length
        vessel_thicker = 1  # TODO: Should this be a settable parameter?
        vessel_length_variance = self.params.vessel_length_variance
        cell_end_thick = self.params.cell_end_thick

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
                        self.x_grid_all[*(*indx_vessel[i_vessel].T, i_slice)],
                        self.y_grid_all[*(*indx_vessel[i_vessel].T, i_slice)]
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

    def generate_raycell(self, ray_idx: int, ray_width: npt.NDArray, input_volume: npt.NDArray):
        """Generate ray cell"""
        logger.info('=' * 80)
        logger.info('Generating ray cell...')
        vol_img_ref_final = np.copy(input_volume)
        ray_cell_length = self.params.ray_cell_length
        ray_cell_variance = self.params.ray_cell_variance
        ray_height = self.params.ray_height
        rcl_d3 = ray_cell_length / 3
        rcl_t2 = ray_cell_length * 2

        # x_grid_all = self.x_grid_all
        # y_grid_all = self.y_grid_all
        # thickness_all = self.thickness_all

        sie_x, _, sie_z = self.params.size_im_enlarge
        x_grid = self.params.x_grid

        cell_end_thick = self.params.cell_end_thick
        # cell_thick = self.params.cell_wall_thick

        for column_idx in ray_idx:
            # between first column and second column need a deviation
            ray_column_rand = 1 / 2 * ray_height
            vessel_end_loc = []
            vessel_end_loc.append(-ray_cell_length * np.random.rand(1))
            lim = sie_x + ray_cell_length
            while vessel_end_loc[-1] < lim:
                temp = ray_cell_length + ray_cell_variance * np.random.randn(1)
                if temp < rcl_d3 or temp > rcl_t2:  # TODO: should this be a clip left_right instead of clip right_right?
                    temp = rcl_t2
                vessel_end_loc.append(vessel_end_loc[-1] + temp)
            vessel_end_loc = np.round(vessel_end_loc)
            vessel_end_loc = vessel_end_loc[vessel_end_loc <= sie_x + ray_cell_length / 2]
            vessel_end_loc = vessel_end_loc.astype(int)

            logger.debug('  Ray cell: %d/%d (cidx, len(ray_idx))', column_idx, len(ray_idx))

            for row_idx in range(len(vessel_end_loc) - 1):
                logger.debug('  Ray cell: %d/%d (ridx, len(vessel_end_loc))', row_idx, len(vessel_end_loc))
                vel_r = int(vessel_end_loc[row_idx])
                vel_r1 = int(vessel_end_loc[row_idx + 1])
                for m2, j_slice in enumerate(np.round(ray_width)):
                    if column_idx % 2 == 0:
                        k = np.round(j_slice + np.round(ray_column_rand))
                    else:
                        k = j_slice

                    tmp_2 = np.round((m2 % 2) * ray_cell_length / 2)
                    vel_col_r = int(vel_r + tmp_2)
                    if vel_col_r < 1:
                        continue
                    vel_col_r1 = int(vel_r1 + tmp_2)
                    if vel_col_r1 > sie_x - 1:
                        continue
                    # vessel_end_loc_column = vessel_end_loc + np.round((m2 % 2) * ray_cell_length / 2)
                    # if vessel_end_loc_column[row_idx+1] > sie_x - 1 or vessel_end_loc_column[row_idx] < 1:
                    #     continue
                    tmp_1 = (np.max((1, k)) + np.min((k + ray_height, sie_z))) / 2
                    t = int(np.round(tmp_1)) - 1
                    if t < 0 or t >= sie_z - 11:
                        continue

                    dx = np.arange(sie_x)
                    x_grid_t = self.x_grid_all[:, t].reshape(x_grid.shape)
                    y_grid_t = self.y_grid_all[:, t].reshape(x_grid.shape)
                    thick_grid_t = self.thickness_all[:, t].reshape(x_grid.shape)

                    # print('x_grid_t:', t, x_grid_t[:, column_idx])
                    try:
                        y_interp1_c = CubicSpline(x_grid_t[:, column_idx], y_grid_t[:, column_idx])(dx) - 1.5
                        thick_interp_c = CubicSpline(x_grid_t[:, column_idx], thick_grid_t[:, column_idx])(dx)
                        y_interp2_c = CubicSpline(x_grid_t[:, column_idx + 1], y_grid_t[:, column_idx + 1])(dx) + 1.5
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

                    cell_neigh_pt = np.array([
                        [vel_col_r, vel_col_r1],
                        [np.round(y_interp1_c[vel_col_r]), np.round(y_interp2_c[vel_col_r])],
                        [np.max((0, k)), np.min((k + ray_height, sie_x))]
                    ])

                    if row_idx == 0:
                        valid_idx = np.arange(
                            vel_col_r + int(cell_end_thick),
                            vel_col_r1 - cell_end_thick // 2 + 1  #Right inclusive
                        )
                    else:
                        valid_idx = np.arange(
                            vel_col_r + cell_end_thick // 2 - 1,
                            vel_col_r1 - cell_end_thick // 2 + 1  #Right inclusive
                        )
                    valid_idx = set(int(_) for _ in valid_idx)

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

                        if idx in valid_idx:
                            # print(idx, valid_idx)
                            for j in range(int(cell_neigh_pt[1, 0]), int(cell_neigh_pt[1, 1]) + 1):
                                for s in range(int(cell_neigh_pt[2, 0]), int(cell_neigh_pt[2, 1]) + 1):
                                    inner_elipse = (
                                        (j - cell_center[idx, 1])**2 / (cell_r[idx, 0] - thick_interp_c[idx])**2 +
                                        (s - cell_center[idx, 2])**2 / (cell_r[idx, 1] - thick_interp_c[idx])**2
                                    )

                                    outer_elipse = (
                                        (j - cell_center[idx, 1])**2 / cell_r[idx, 0]**2 +
                                        (s - cell_center[idx, 2])**2 / cell_r[idx, 1]**2
                                    )

                                    if outer_elipse < 1:
                                        # print('   ', j, s)
                                        vol_img_ref_final[idx, j, s] = int(
                                            (1 / (1 + np.exp(-(inner_elipse - 1) / .05))) * 255
                                        )
        return vol_img_ref_final

    def generate_deformation(self, ray_cell_idx: npt.NDArray, idx_skip_all: npt.NDArray, idx_vessel_cen: npt.NDArray):
        """Add complicated deformation to the volume image. The deformation fields are generated separately.
        Then, they are summed together. Here u, v are initialized to be zero. Then they are summed."""
        logger.info('=' * 80)
        logger.info('Generating deformation...')
        sie_x, sie_y, _ = self.params.size_im_enlarge

        x_vector = self.params.x_vector
        y_vector = self.params.y_vector
        lx = len(x_vector)
        ly = len(y_vector)

        idx_skip_all = set(int(_) for _ in idx_skip_all.flatten())

        x,y = np.meshgrid(np.arange(sie_x), np.arange(sie_y), indexing='ij')
        u = np.zeros_like(x, dtype=float)
        v = np.zeros_like(x, dtype=float)
        u1 = np.zeros_like(x, dtype=float)
        v1 = np.zeros_like(x, dtype=float)

        for i in range(1, lx-1, 2):
            logger.debug('  i: %d/%d', i, lx-1)
            for j in range(1, ly-1, 2):
                i1 = i + ((j+1) % 4 == 0)
                idx = i1 * ly + j
                if idx in idx_skip_all:
                    continue
                # print(f'  j: {j} / {ly}')
                mm = np.min(abs(j - ray_cell_idx))
                is_close_to_ray = mm <= 4
                is_close_to_ray_far = mm <= 8

                vessel_idx = np.where(idx == idx_vessel_cen)[0]
                pcx, pcy = self.x_grid_all[idx, 0], self.y_grid_all[idx, 0]
                if len(vessel_idx) == 0:
                    # Small fibers
                    if is_close_to_ray:
                        k = [0.08, 0.06, 2 + np.random.rand(), 0.3 * (1 + np.random.rand())]
                    else:
                        k = [0.08, 0.06, 2 + np.random.rand(), 1.0 * (1 + np.random.rand())]
                    u_temp = local_distort(x,y, pcx, pcy, k)

                    if is_close_to_ray:
                        k = [0.08, 0.06, 2 + np.random.rand(), 0.3 * (1 + np.random.rand())]
                    else:
                        k = [0.08, 0.06, 2 + np.random.rand(), 1.0 * (1 + np.random.rand())]
                    v_temp = local_distort(y,x, pcy, pcx, k)
                    sign_temp = np.sign(np.random.randn())

                    u += -sign_temp * u_temp
                    v += -sign_temp * v_temp

                else:
                    # Large vessels
                    # ----------the value here are super important-------------
                    if is_close_to_ray_far:
                        k = [0.06, 0.055, 2, 3 + 5 * np.random.rand()]
                    else:
                        k = [0.06, 0.055, 2, 15]

                    u += local_distort(x,y, pcx, pcy, k)
                    v += local_distort(y,x, pcy, pcx, k)

                if np.random.rand() < 0.01:
                    # ----------the value here are super important-------------
                    if is_close_to_ray_far:
                        k = [0.01, 0.008, 1.5 * (1 + np.random.rand(1)), 0.2 * (1 + np.random.rand(1))]
                    else:
                        k = [0.01, 0.008, 1.5 * (1 + np.random.rand(1)), 0.5 * (1 + np.random.rand(1))]
                    if np.random.randn() > 0:
                        u1 += np.sign(np.random.randn()) * local_distort(x,y, pcx, pcy, k)
                    else:
                        v1 += np.sign(np.random.randn()) * local_distort(y,x, pcy, pcx, k)

        return u, v, u1, v1

    def ray_cell_shrinking(self, width: npt.NDArray, idx_all: npt.NDArray, dist_v: npt.NDArray) -> npt.NDArray:
        """Shrink the ray cell width"""
        logger.info('=' * 80)
        logger.info('Ray cell shrinking...')
        grid_shape = self.params.x_grid.shape
        # y_vector = self.params.y_vector
        x_grid_all = self.x_grid_all
        y_grid_all = self.y_grid_all
        # thickness_all = self.thickness_all

        cell_thick = self.params.cell_wall_thick
        sie_x, sie_y, sie_z = self.params.size_im_enlarge

        ray_height = self.params.ray_height
        ray_size = self.params.cell_r - cell_thick / 2
        slice_idx = self.params.save_slice
        # TODO: need to implement slice_idx with multiple indexes
        # I think this was reused from some other code because when the output array
        # is used it assumes it does not have a third dimension (or that it is always 1)

        # thick_left = thickness_all[:, slice_idx]
        x_node_grid = x_grid_all[:, slice_idx].reshape(*grid_shape)
        y_node_grid = y_grid_all[:, slice_idx].reshape(*grid_shape)
        logger.debug('x_node_grid.shape: %s', x_node_grid.shape)
        # print('y_node_grid.shape:', y_node_grid.shape)
        # thick_node_grid = thick_left.reshape(*grid_shape)

        _, y_grid = np.mgrid[0:sie_x, 0:sie_y]

        cnt = defaultdict(int)
        for idx in idx_all.flatten():
            cnt[int(idx)] += 1
        # idx_all = set(int(_) for _ in idx_all.flatten())

        base_k = 0
        dx = np.arange(sie_x, dtype=int)
        v_all = np.zeros((sie_x, sie_y), dtype=float)

        for key in cnt.keys():
            coeff1 = np.ones(sie_z)
            coeff2 = np.zeros(sie_z)
            v1 = np.zeros_like(y_grid, dtype=float)
            v2 = np.zeros_like(y_grid, dtype=float)

            # indicator = []
            v1_all = None
            v2_all = None

            logger.debug('  Ray cell shrinking: key = %d  cnt[key] = %d', key, cnt[key])
            # TODO: check does this relies on the fact that idx_all should be ordered?
            for key_cnt in range(cnt[key]):
                k = base_k + key_cnt
                idx = idx_all[k]

                # print('  ', idx, x_node_grid[:, idx].shape, y_node_grid[:, idx].shape, dx.shape)
                logger.debug(f'  {idx=} {x_node_grid[:, idx].shape=} {y_node_grid[:, idx].shape=} {dx.shape=}')
                y_node_grid_1 = CubicSpline(x_node_grid[:, idx], y_node_grid[:, idx])(dx)
                y_node_grid_2 = CubicSpline(x_node_grid[:, idx + 2], y_node_grid[:, idx + 2])(dx)

                # indv1 = (dx * sie_y + np.round(y_node_grid_1)).astype(int)
                # indv2 = (dx * sie_y + np.round(y_node_grid_2)).astype(int)

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
            v_all[:, :] += coeff1[slice_idx] * v1_all + coeff2[slice_idx] * v2_all

        return v_all

    def apply_deformation(self, vol_img_ref: npt.NDArray, u: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
        """Apply the deformation to the volume image"""
        logger.info('=' * 80)
        logger.info('Applying deformation...')
        sie_x, sie_y, _ = self.params.size_im_enlarge
        x_grid, y_grid = np.mgrid[0:sie_x, 0:sie_y]
        x_interp = x_grid + u
        y_interp = y_grid + v

        Vq = griddata(
            (x_interp.flatten(), y_interp.flatten()),
            vol_img_ref[:,:,self.params.save_slice].flatten(),
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
        save_slice = self.params.save_slice
        filename = os.path.join(self.root_dir, dirname, f'volImgRef_{save_slice+1:05d}.tiff')

        logger.debug('Saving slice %d to %s', save_slice, filename)
        logger.debug('vol_img_ref.shape: %s', vol_img_ref.shape)
        logger.debug('min/max: %f %f', np.min(vol_img_ref), np.max(vol_img_ref))

        self.save_2d_img(vol_img_ref[:, :, save_slice], filename)

    @staticmethod
    def save_2d_img(data: npt.NDArray, filename: str):
        """Save 2D data to a TIFF file"""
        # print(np.where(np.isnan(data)))
        img = Image.fromarray(data.astype(np.uint8), mode='L')
        img.show()
        img.save(filename)

    def save_distortion(self, u: npt.NDArray, v: npt.NDArray):
        """Save the distortion fields"""
        u_name = os.path.join(self.root_dir, 'LocalDistVolumeDispU', f'u_volImgRef_{self.params.save_slice:05d}.csv')
        v_name = os.path.join(self.root_dir, 'LocalDistVolumeDispV', f'v_volImgRef_{self.params.save_slice:05d}.csv')
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

        # ray_cell_x_ind, ray_cell_width, keep_ray_cell, ray_cell_x_ind_all_update = self.distrbute_ray_cells(ray_cell_x_ind_all)
        ray_cell_x_ind, ray_cell_width, ray_cell_x_ind_all_update = self.distrbute_ray_cells(ray_cell_x_ind_all)
        # print('ray_cell_x_ind:', ray_cell_x_ind.shape, ray_cell_x_ind)
        # print('ray_cell_width:',)
        logger.debug('ray_cell_x_ind: %s  %s', ray_cell_x_ind.shape, ray_cell_x_ind)
        logger.debug('ray_cell_width:')

        for i,width in enumerate(ray_cell_width):
            logger.debug('   %d %s', i+1, width)
        # logger.debug('keep_ray_cell: %s  %s', keep_ray_cell.shape, keep_ray_cell)
        logger.debug('ray_cell_x_ind_all_update: %s  %s', ray_cell_x_ind_all_update.shape, ray_cell_x_ind_all_update)
        # sys.exit(0)

        vol_img_ref = 255 * np.ones(self.params.size_im_enlarge)
        # vol_img_ref = self.generate_small_fibers(ray_cell_x_ind_all, keep_ray_cell, indx_skip_all, vol_img_ref)
        vol_img_ref = self.generate_small_fibers(ray_cell_x_ind, indx_skip_all, vol_img_ref)
        vol_img_ref = self.generate_large_fibers(indx_vessel, indx_vessel_cen, indx_skip_all, vol_img_ref)

        if self.params.is_exist_ray_cell:
            for idx, width in zip(ray_cell_x_ind, ray_cell_width):
                logger.debug('Generating ray cell: %s / %s', idx, width)
                vol_img_ref = self.generate_raycell(idx, width, vol_img_ref)

        # Save the generated volume
        self.create_dirs()
        self.save_slice(vol_img_ref, 'volImgBackBone')

        # sys.exit(0)
        # u1 and v1 are in a commented part of the code. Prob used in original code?
        u, v, _, _ = self.generate_deformation(ray_cell_x_ind, indx_skip_all, indx_vessel_cen)
        logger.debug('u.shape: %s  min/max: %s %s', u.shape, u.min(), u.max())
        logger.debug('v.shape: %s  min/max: %s %s', v.shape, v.min(), v.max())

        np.save('x_grid_all.npy', self.x_grid_all)
        np.save('y_grid_all.npy', self.y_grid_all)
        np.save('vol_img_ref.npy', vol_img_ref)
        np.save('u.npy', u)
        np.save('v.npy', v)
        if self.params.is_exist_ray_cell:
            v_all_ray = self.ray_cell_shrinking(ray_cell_width, ray_cell_x_ind_all_update, v)
            v += v_all_ray[:,:]
            logger.debug('vray   : %s  min/max: %s %s', v_all_ray.shape, v_all_ray.min(), v_all_ray.max())
            logger.debug('v.shape: %s  min/max: %s %s', v.shape, v.min(), v.max())
            np.save('v_all_ray.npy', v_all_ray)
        else:
            v_all_ray = np.zeros_like(v)
            np.save('v_all_ray.npy', v_all_ray)

        self.save_distortion(u, v)

        img_interp = self.apply_deformation(vol_img_ref, u, v)

        np.save('img_interp.npy', img_interp)

        filename = os.path.join(self.root_dir, 'LocalDistVolume', f'volImgRef_{self.params.save_slice+1:05d}.tiff')
        self.save_2d_img(img_interp, filename)

        logger.info('======== DONE ========')
