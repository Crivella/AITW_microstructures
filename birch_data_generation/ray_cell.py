"""Ray cells"""
import os
import time

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.interpolate import CubicSpline

from .fit_elipse import fit_elipse
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

        grid_shape = self.params.x_grid.shape
        grid_size = self.params.x_grid.size

        x_grid = self.params.x_grid.flatten()
        y_grid = self.params.y_grid.flatten()

        # x_grid_interp = np.empty((grid_size, len(slice_interest)))
        # y_grid_interp = np.empty((grid_size, len(slice_interest)))
        # thickness_interp = np.empty((grid_size, len(slice_interest)))
        # for t, i_slice in enumerate(slice_interest):
        #     x_grid_interp[:, t] = x_grid + np.random.randn(grid_size) * 3 - 1.5
        #     y_grid_interp[:, t] = y_grid + np.random.randn(grid_size) * 3 - 1.5
        #     thickness_interp[:, t] = self.params.cell_wall_thick - 0.5 + np.random.randn(grid_size)
        x_grid_interp = np.random.rand(grid_size, l) * 3 - 1.5 + x_grid[:, np.newaxis]
        y_grid_interp = np.random.rand(grid_size, l) * 3 - 1.5 + y_grid[:, np.newaxis]
        thickness_interp = np.random.rand(grid_size, l) + self.params.cell_wall_thick - 0.5

        # TODO: redo this with proper broadcasting and use RegularGridInterpolator
        x_grid_interp = x_grid_interp.reshape(-1, l)
        y_grid_interp = y_grid_interp.reshape(-1, l)
        thickness_interp = thickness_interp.reshape(-1, l)

        interp_x = np.arange(1, self.params.size_im_enlarge[2] + 1)
        x_grid_all = np.empty((grid_size, len(interp_x)))
        y_grid_all = np.empty((grid_size, len(interp_x)))
        thickness_all = np.empty((grid_size, len(interp_x)))
        # print(self.params.size_im_enlarge)
        # print(slice_interest, slice_interest.shape)
        # print(x_grid.shape, x_grid.size)
        # print(x_grid_all.shape)
        # print(x_grid_interp.shape)
        # print(interp_x.shape)
        for i in range(x_grid.size):
            cs_x = CubicSpline(slice_interest, x_grid_interp[i, :])
            cs_y = CubicSpline(slice_interest, y_grid_interp[i, :])
            cs_t = CubicSpline(slice_interest, thickness_interp[i, :])
            x_grid_all[i, :] = cs_x(interp_x)
            y_grid_all[i, :] = cs_y(interp_x)
            thickness_all[i, :] = cs_t(interp_x)

        self.x_grid_all = x_grid_all
        self.y_grid_all = y_grid_all
        self.thickness_all = thickness_all

        return x_grid_all, y_grid_all, thickness_all

    def get_ray_cell_indexes(self) -> npt.NDArray:
        """Get ray cell indexes"""
        ray_cell_x_ind_all = np.empty((1, 0))
        if self.params.is_exist_ray_cell:
            ray_cell_linspace = np.arange(10, len(self.params.y_vector) - 9, self.params.ray_space)
            ray_cell_x_ind_all = ray_cell_linspace + np.random.rand(len(ray_cell_linspace)) * 10 - 5
            ray_cell_x_ind_all = ray_cell_x_ind_all // 2 + 1

        return ray_cell_x_ind_all

    def get_vessels_all(self, ray_cell_x_ind_all: npt.NDArray = None):
        """Get vessels"""
        if not self.params.is_exist_vessel:
            return np.empty((0, 2))
        x_vector = self.params.x_vector
        y_vector = self.params.y_vector

        x_rand_1 = np.round(np.random.rand(self.params.vessel_count) * (len(x_vector) - 16) + 8) / 2 * 2 - 1
        y_rand_1 = np.round(np.random.rand(self.params.vessel_count) * (len(y_vector) - 14) + 7) / 4 * 4
        y_rand_2 = np.round(np.random.rand(self.params.vessel_count // 2) * (len(y_vector) - 14) + 7) / 2 * 2
        x_rand_2 = np.round(np.random.rand(self.params.vessel_count // 2) * (len(x_vector) - 16) + 8) / 2 * 2

        x_rand_all = np.concatenate((x_rand_1, x_rand_2))
        y_rand_all = np.concatenate((y_rand_1, y_rand_2))
        vessel_all = np.column_stack((x_rand_all, y_rand_all))

        # Remove some vessel that too close to the other vessels
        vessel_all = self.vessel_filter_close(vessel_all)
        vessel_all = self.vessel_filter_ray_close(vessel_all, ray_cell_x_ind_all)
        vessel_all = self.vessel_extend(vessel_all)
        vessel_all = self.vessel_filter_ray_close(vessel_all, ray_cell_x_ind_all)

        return vessel_all

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
        vessel_all_extend = np.empty((0, 3))
        for vessel in vessel_all:
            dist = vessel_all - vessel

            mark0 = np.where((dist[:, 0] <= 24) & (dist[:, 0] >= -8) & np.abs(dist[:, 1]) <= 8)[0]
            mark1 = np.where((dist[:, 0] <= 12) & (dist[:, 0] >= -6) & np.abs(dist[:, 1]) <= 6)[0]

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
        lx = len(self.params.x_vector)
        lv = len(vessel_all)

        indx_skip_all = np.empty((lv, 6, 2))
        indx_skip_all[:, :, :] = vessel_all[:, np.newaxis, :]
        indx_skip_all += [
            (-1, -2),
            (+1, -2),
            (-2, +0),
            (+2, +0),
            (-1, +2),
            (+1, +2)
        ]

        indx_vessel = np.empty((lv, 6, 2))
        indx_vessel[:, :, :] = vessel_all[:, np.newaxis, :]
        indx_vessel += [
            (-3, -1),
            (-3, +1),
            (+0, -3),
            (+0, +3),
            (+3, -1),
            (+3, +1)
        ]

        indx_vessel_cen = vessel_all[:, 0] * lx + vessel_all[:, 1]
        indx_vessel_cen = indx_vessel_cen.astype(int)

        return indx_skip_all, indx_vessel, indx_vessel_cen

    def distrbute_ray_cells(self, ray_cell_x_ind_all: npt.NDArray):
        """Distribute the ray cells across the volume"""
        x_ind = []
        width = []
        keep = []
        x_ind_all_update = []
        print(ray_cell_x_ind_all)
        # if self.params.is_exist_ray_cell:

        sie_z = self.params.size_im_enlarge[2]

        ray_cell_num = self.params.ray_cell_num
        ray_cell_num_std = self.params.ray_cell_num_std
        ray_height = self.params.ray_height

        m = sie_z / ray_cell_num / ray_height + 6
        for i, idx in enumerate(ray_cell_x_ind_all):
            app = [0]
            ray_cell_space = np.round(16 * np.random.rand(int(np.ceil(m))) + 6)
            rnd = np.round(-30 * np.random.rand())
            for rs in ray_cell_space:
                ray_idx = [idx, idx + 1]
                group = max(5, min(25,
                    np.round(np.random.randn() * ray_cell_num_std + ray_cell_num)
                ))
                # print('rs,ray_cell, group:', rs, ray_cell_space, group)
                # print('app:', app)
                app = app[-1] + (np.arange(group + 1) + rs + rnd) * ray_height
                rnd = 0

                if app[0] <= sie_z - 150 and app[-1] >= 150:
                    x_ind.append(ray_idx)
                    x_ind_all_update.append(idx)
                    width.append(np.round(app))
                    keep.append(i)

        # print('-'*80)
        # print(x_ind)
        # for w in width:
        #     print(len(w))
        # print(width)
        # print(keep)
        # print(x_ind_all_update)
        # print('-'*80)

        return (
            np.array(x_ind, dtype=int),
            width,
            # np.array(width, dtype=float),
            np.array(keep, dtype=int),
            np.array(x_ind_all_update, dtype=int)
        )

    def generate_small_fibers(
            self,
            x_ind: npt.NDArray,
            keep: npt.NDArray,
            # ray_idx: npt.NDArray,
            indx_skip_all: npt.NDArray,
            input_volume: npt.NDArray
        ) -> npt.NDArray:
        """Generate small fibers."""
        vol_img_ref = np.copy(input_volume)

        skip_fiber_column = np.concatenate((
            x_ind[keep],
            x_ind[keep] + 1
        ))
        skip_fiber_column = set(int(_) for _ in skip_fiber_column)


        print('skip_fiber_column:', skip_fiber_column)
        sie_x, sie_y, sie_z = self.params.size_im_enlarge

        x_vector = self.params.x_vector
        y_vector = self.params.y_vector

        neigh_loc = self.params.neighbor_local

        cell_length = self.params.cell_length
        cell_length_variance = self.params.cell_length_variance
        cell_end_thick = self.params.cell_end_thick

        for i in range(1, len(x_vector)-1, 2):
            for j in range(1, len(y_vector)-1, 2):
                if j in skip_fiber_column:
                    continue

                print(i,j)

                # is_close_to_ray = False
                # if np.min(np.abs(j - ray_idx)) <= 4:
                #     is_close_to_ray = True
                # is_close_to_ray_far = False
                # if np.min(np.abs(j - ray_idx)) <= 8:
                #     is_close_to_ray_far = True

                initial = np.round(np.random.rand() * cell_length)
                fiber_end_loc_all = [initial,]

                for s in range(int(np.ceil(sie_z / cell_length)) + 7):
                    # temp = np.min((
                    #     3 * cell_length,
                    #     np.max((100, cell_length + np.random.randn() * cell_length_variance))
                    # ))
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
                print('fiber_end:', fiber_end)

                # The arrangement of the cells should be staggered. So every four nodes,
                # they should deviate along x direction.
                i1 = i + (j % 4 == 0)

                idx = len(x_vector) * j + i1
                if np.any(indx_skip_all == idx):
                    # if this node exist on the surface of vessels, skip it.
                    continue

                # Skip some fibers
                skip_cell_thick = 0  # TODO: Should this be a settable parameter?
                for i_slice in range(sie_z):
                    if i_slice in fiber_end:
                        # if this slice is not the end of this cell inside the lumen, it should be black
                        continue
                    # used to store the four neighbored points for elipse fitting
                    point_coord = np.empty((4, 2))

                    for k in range(4):
                        neigh_idx = len(x_vector) * (j + neigh_loc[1, k]) + i1 + neigh_loc[0, k]
                        point_coord[k, :] = [self.x_grid_all[neigh_idx, i_slice], self.y_grid_all[neigh_idx, i_slice]]

                    # make the elipse more elipse
                    if skip_cell_thick == 0:
                        point_coord[1, 1] -= 2
                        point_coord[3, 1] += 2

                    r1, r2, h, k = fit_elipse(point_coord)  # Estimate the coefficients of the ellipse.

                    # start = time.time()
                    # Then we can estimate the diameter along two direction for the fiber
                    # The rectangle region covering the ellipse.
                    mr = np.floor(max(r1, r2))
                    # TODO: check if it is normal to obtain NaNs here and if matlab is also just ignoring them
                    if np.isnan(mr):
                        continue
                    # print('mr: ', mr)
                    region_cell_ind_x = int(np.ceil(h)) + np.arange(-mr, mr + 1)
                    region_cell_ind_y = int(np.ceil(k)) + np.arange(-mr, mr + 1)
                    region_cell_ind_x = np.arange(
                        np.max((1, np.min(region_cell_ind_x))),
                        np.min((sie_x, np.max(region_cell_ind_x)))
                    )
                    region_cell_ind_y = np.arange(
                        np.max((1, np.min(region_cell_ind_y))),
                        np.min((sie_y, np.max(region_cell_ind_y)))
                    )

                    # nxt = time.time()
                    # print('time_after_fitting1:', nxt - start)
                    # start = nxt

                    if len(region_cell_ind_x) == 0 or len(region_cell_ind_y) == 0:
                        continue

                    region_cell_x, region_cell_y = np.meshgrid(region_cell_ind_x, region_cell_ind_y, indexing='ij')

                    # # External contour of the elipse
                    # in_elipse1 = (region_cell_x - h)**2 / r1**2 + (region_cell_y - k)**2 / r2**2
                    # Internal contour of the elipse
                    in_elipse2 = (
                        (region_cell_x - h)**2 / (r1 - self.thickness_all[idx, i_slice] - skip_cell_thick)**2 +
                        (region_cell_y - k)**2 / (r2 - self.thickness_all[idx, i_slice] - skip_cell_thick)**2
                    )

                    # nxt = time.time()
                    # print('time_after_fitting2:', nxt - start)
                    # start = nxt

                    region_cell = 1 / (1 + np.exp(-(in_elipse2 - 1) / 0.05))

                    # nxt = time.time()
                    # print('time_after_fitting3:', nxt - start)
                    # start = nxt

                    # print(region_cell_ind_x.shape, region_cell_ind_y.shape)
                    # print(region_cell.shape)

                    x_idx, y_idx = np.meshgrid(region_cell_ind_x, region_cell_ind_y, indexing='ij')
                    region_cell_ind_x = x_idx.astype(int)
                    region_cell_ind_y = y_idx.astype(int)
                    region_cell = region_cell

                    vol_img_ref[region_cell_ind_x, region_cell_ind_y, i_slice] *= region_cell

                    # print('time_after_fitting4:', time.time() - start)
                    # print()

        vol_img_ref = np.astype(vol_img_ref, int)
        vol_img_ref = np.clip(vol_img_ref, 0, 255)

        return vol_img_ref

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
        vol_img_ref = np.copy(input_volume)

        x_vector = self.params.x_vector
        y_vector = self.params.y_vector

        size_im_enlarge = self.params.size_im_enlarge
        sie_x = size_im_enlarge[0]
        sie_y = size_im_enlarge[1]
        sie_z = size_im_enlarge[2]

        vessel_length = self.params.vessel_length
        vessel_thicker = 1  # TODO: Should this be a settable parameter?
        vessel_length_variance = self.params.vessel_length_variance
        cell_end_thick = self.params.cell_end_thick


        for i in range(1, len(x_vector)-1, 2):
            for j in range(1, len(y_vector)-1, 2):
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

                # The arrangement of the cells should be staggered. So every four nodes,
                # they should deviate along x direction.
                i1 = i + (j % 4 == 0)
                idx = len(x_vector) * j + i1

                for i_slice in range(sie_z):
                    i_vessel = np.where(idx == indx_vessel_cen)[0]
                    if not i_vessel.size:
                        continue

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
                                vol_img_ref[t1, t2, i_slice] = 1 / (1 + np.exp(-(in_elipse2 - 1) / 0.05)) * 255
        return vol_img_ref

    def generate_raycell(self, ray_idx: int, ray_width: npt.NDArray, input_volume: npt.NDArray):
        """Generate ray cell"""
        vol_img_ref_final = np.copy(input_volume)
        ray_cell_length = self.params.ray_cell_length
        ray_cell_variance = self.params.ray_cell_variance
        ray_height = self.params.ray_height
        rcl_d3 = ray_cell_length / 3
        rcl_t2 = ray_cell_length * 2

        x_grid_all = self.params.x_grid_all
        y_grid_all = self.params.y_grid_all
        thickness_all = self.params.thickness_all

        sie_x, sie_y, sie_z = self.params.size_im_enlarge
        x_grid = self.params.x_grid

        cell_end_thick = self.params.cell_end_thick
        cell_thick = self.params.cell_wall_thick

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

            for row_idx in range(len(vessel_end_loc) - 1):
                for m2, j_slice in enumerate(np.round(ray_width[row_idx])):
                    if column_idx % 2 == 0:
                        k = np.round(j_slice + np.round(ray_column_rand))
                    else:
                        k = j_slice
                    app = (np.max((1, k)) + np.min((k + ray_height, sie_x))) / 2
                    t = int(np.round(app)) - 1
                    if t < 0 or t >= sie_z - 11:
                        continue
                    vessel_end_loc_column = vessel_end_loc + np.round(np.mod(m2, 2) * ray_cell_length / 2)
                    if vessel_end_loc_column[row_idx+1] > sie_x - 1 or vessel_end_loc_column[row_idx] < 2:
                        continue

                    dx = np.arange(1, sie_x+1)
                    x_grid_t = self.x_grid_all[:, t].reshape(x_grid.shape)
                    y_grid_t = self.y_grid_all[:, t].reshape(x_grid.shape)
                    thick_grid_t = self.thickness_all[:, t].reshape(x_grid.shape)

                    y_interp1_c = CubicSpline(x_grid_t[:, column_idx], y_grid_t[:, column_idx])(dx) - 1.5
                    thick_interp_c = CubicSpline(x_grid_t[:, column_idx], thick_grid_t[:, column_idx])(dx)
                    y_interp2_c = CubicSpline(x_grid_t[:, column_idx + 1], y_grid_t[:, column_idx + 1])(dx) + 1.5

                    cell_center = np.column_stack((
                        dx,
                        np.round((y_interp2_c + y_interp1_c) / 2),
                        np.full(dx.shape, app)
                    ))
                    cell_r = np.column_stack((
                        (y_interp2_c - y_interp1_c) / 2,
                        np.full(dx.shape, (np.min((k + ray_height, sie_x)) - np.max((1, k))) / 2)
                    )) + 0.5

                    velc_idx = vessel_end_loc_column[row_idx]
                    velc_idx1 = vessel_end_loc_column[row_idx + 1]
                    cell_neigh_pt = np.array([
                        [velc_idx, velc_idx1],
                        [np.round(y_interp1_c[velc_idx]), np.round(y_interp2_c[velc_idx])],
                        [np.max((1, k)), np.min((k + ray_height, sie_x))]
                    ])

                    if row_idx == 0:
                        valid_idx = np.arange(
                            vessel_end_loc[row_idx] + int(cell_end_thick),
                            vessel_end_loc[row_idx + 1] - cell_end_thick // 2 + 1  #Right inclusive
                        )
                    else:
                        valid_idx = np.arange(
                            vessel_end_loc[row_idx] + cell_end_thick // 2 - 1,
                            vessel_end_loc[row_idx + 1] - cell_end_thick // 2 + 1  #Right inclusive
                        )

                    for idx in range(vessel_end_loc[row_idx] + 1, vessel_end_loc[row_idx + 1]+1):
                        if j_slice == np.min(ray_width):
                            vol_img_ref_final[idx, int(y_interp1_c[idx]):int(y_interp2_c[idx] + 1), int(cell_center[idx, 2]):int(cell_neigh_pt[2, 1] + 1)] = 255
                        elif j_slice == np.max(ray_width):
                            vol_img_ref_final[idx, int(y_interp1_c[idx]):int(y_interp2_c[idx] + 1), int(cell_neigh_pt[2, 0]):int(cell_center[idx, 2])] = 255
                        else:
                            vol_img_ref_final[idx, int(y_interp1_c[idx]):int(y_interp2_c[idx] + 1), int(cell_neigh_pt[2, 0]):int(cell_neigh_pt[2, 1])] = 255

                        if np.any(valid_idx == idx):
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
                                        vol_img_ref_final[idx, j, s] = int(
                                            (1 / (1 + np.exp(-(inner_elipse - 1) / .05))) * 255
                                        )
        return vol_img_ref_final

    def generate_deformation(self, ray_cell_idx: npt.NDArray, idx_skip_all, idx_vessel_cen):
        """Add complicated deformation to the volume image. The deformation fields are generated separately.
        Then, they are summed together. Here u, v are initialized to be zero. Then they are summed."""
        sie_x, sie_y, sie_z = self.params.size_im_enlarge

        x_vector = self.params.x_vector
        y_vector = self.params.y_vector

        x,y = np.meshgrid(np.arange(sie_x), np.arange(sie_y), indexing='ij')
        u = np.zeros_like(x)
        v = np.zeros_like(x)
        u1 = np.zeros_like(x)
        v1 = np.zeros_like(x)

        t = 0
        for i in range(1, len(x_vector), 2):
            for j in range(1, len(y_vector), 2):
                t += 1
                mm = np.min(abs(j - ray_cell_idx))
                is_close_to_ray = mm <= 4
                is_close_to_ray_far = mm <= 8

                i1 = i + (j % 4 == 0)

                idx = len(x_vector) * j + i1
                if np.any(idx_skip_all == idx):
                    continue

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
                        k = [0.01, 0.008, 1.5 * (1 + np.random.rand(1)), 0.2 * (1 + 1 * np.random.rand(1))]
                    else:
                        k = [0.01, 0.008, 1.5 * (1 + np.random.rand(1)), 0.5 * (1 + 1 * np.random.rand(1))]
                    if np.random.randn() > 0:
                        u1 += np.sign(np.random.randn()) * local_distort(x,y, pcx, pcy, k)
                    else:
                        v1 += np.sign(np.random.randn()) * local_distort(y,x, pcy, pcx, k)

        return u, v, u1, v1

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
        save_slice = self.params.save_slice - 1
        filename = os.path.join(self.root_dir, dirname, f'volImgRef_{save_slice:05d}.tif')

        img = Image.fromarray(vol_img_ref[:, :, save_slice].astype(np.uint8))
        img.save(filename)

    def save_distortion(self, u: npt.NDArray, v: npt.NDArray):
        """Save the distortion fields"""
        u_name = os.path.join(self.root_dir, 'LocalDistVolumeDispU', f'u_volImgRef_{self.params.save_slice:05d}.csv')
        v_name = os.path.join(self.root_dir, 'LocalDistVolumeDispV', f'v_volImgRef_{self.params.save_slice:05d}.csv')
        np.savetxt(u_name, np.round(u[:, :, self.params.save_slice], decimals=4), delimiter=',')
        np.savetxt(v_name, np.round(v[:, :, self.params.save_slice], decimals=4), delimiter=',')

    def apply_distortion(self, vol_img_ref: npt.NDArray, u: npt.NDArray, v: npt.NDArray):
        raise NotImplementedError('Function not implemented yet')

    def generate(self):
        """Generate ray cells"""
        np.random.seed(self.params.random_seed)

        self.get_grid_all()

        print(self.params.size_im_enlarge)
        print('x_vector:', self.params.x_vector.shape)
        print('y_vector:', self.params.y_vector.shape)

        print('x_grid_all:', self.x_grid_all.shape)

        ray_cell_x_ind_all = self.get_ray_cell_indexes()
        print('ray_cell_x_ind_all:', ray_cell_x_ind_all.shape, ray_cell_x_ind_all)
        vessel_all = self.get_vessels_all(ray_cell_x_ind_all)
        print('vessel_all', vessel_all.shape, vessel_all)

        indx_skip_all, indx_vessel, indx_vessel_cen = self.fiber_filter_in_vessel(vessel_all)
        print('indx_skip_all:', indx_skip_all.shape)
        print('indx_vessel:', indx_vessel.shape)
        print('indx_vessel_cen:', indx_vessel_cen.shape)

        vol_img_ref = np.zeros(self.params.size_im_enlarge)

        ray_cell_x_ind, ray_cell_width, keep_ray_cell, ray_cell_x_ind_all_update = self.distrbute_ray_cells(ray_cell_x_ind_all)
        print('ray_cell_x_ind:', ray_cell_x_ind.shape, ray_cell_x_ind)
        print('ray_cell_width:', len(ray_cell_width))
        print('keep_ray_cell:', keep_ray_cell.shape, keep_ray_cell)
        print('ray_cell_x_ind_all_update:', ray_cell_x_ind_all_update.shape)

        vol_img_ref = 255 * np.ones(self.params.size_im_enlarge)
        # vol_img_ref = self.generate_small_fibers(ray_cell_x_ind_all, keep_ray_cell, indx_skip_all, vol_img_ref)
        vol_img_ref = self.generate_large_fibers(indx_vessel, indx_vessel_cen, indx_skip_all, vol_img_ref)

        if self.params.is_exist_ray_cell:
            for idx, width in zip(ray_cell_x_ind, ray_cell_width):
                vol_img_ref = self.generate_raycell(idx, width, vol_img_ref)

        u, v, u1, v1 = self.generate_deformation(ray_cell_x_ind, indx_skip_all, indx_vessel_cen)

        # Save the generated volume
        self.create_dirs()
        self.save_slice(vol_img_ref, 'volImgBackBone')
        self.save_distortion(u, v)
