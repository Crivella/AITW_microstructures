"""Ray cells"""
import os

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from . import vessels as ves
from .microstructure import WoodMicrostructure


class SpruceMicrostructure(WoodMicrostructure):
    save_prefix = 'SaveSpruce'
    local_distortion_cutoff = 200

    def __init__(self, *args, **kwargs):
        """Initialize the SpruceMicrostructure class"""
        super().__init__(*args, **kwargs)

        self.thickness_all_ray = None
        self.thickness_all_fiber = None

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
        thickness_interp_ray = (cwt - 0.5) * np.random.rand(gx, gy, l) + np.random.rand(gx, gy, l)
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

        self.x_grid_all = x_grid_all
        self.y_grid_all = y_grid_all
        self.thickness_all_ray = thickness_all_ray
        self.thickness_all_fiber = thickness_all_fiber

        return x_grid_all, y_grid_all, thickness_all_ray, thickness_all_fiber

    def _generate(self):
        """Generate ray cells"""
        np.random.seed(self.params.random_seed)

        thick_all_valid_sub, compress_all_valid_sub = self.get_distortion_map()
        self.logger.debug('thick_all_valid_sub.shape: %s', thick_all_valid_sub.shape)
        self.logger.debug('compress_all_valid_sub.shape: %s', compress_all_valid_sub.shape)

        self.get_grid_all(thick_all_valid_sub)

        self.logger.debug('PARAM: size_im_enlarge: %s', self.params.size_im_enlarge)
        self.logger.debug('PARAM: x_vector.shape: %s', self.params.x_vector.shape)
        self.logger.debug('PARAM: y_vector.shape: %s', self.params.y_vector.shape)
        self.logger.debug('PARAM: x_grid_all.shape: %s', self.x_grid_all.shape)
        self.logger.debug('PARAM: thickness_all_fiber.shape: %s', self.thickness_all_fiber.shape)

        # ray_cell_x_ind_all = self.get_ray_cell_indexes()
        # self.logger.debug('ray_cell_x_ind_all.shape: %s', ray_cell_x_ind_all.shape)
        # self.logger.debug('ray_cell_x_ind_all: %s', ray_cell_x_ind_all)

        # vessel_all = self.generate_vessel_indexes(ray_cell_x_ind_all)
        # self.logger.debug('vessel_all.shape: %s', vessel_all.shape)
        # # self.logger.debug('vessel_all: %s', vessel_all)

        # indx_skip_all = ves.get_grid_idx_in_vessel(vessel_all)
        # indx_ves_edges = ves.get_grid_idx_edges(vessel_all)
        # indx_vessel_cen = vessel_all
        # self.logger.debug('indx_skip_all: %s', indx_skip_all.shape)
        # self.logger.debug('indx_vessel: %s', indx_ves_edges.shape)
        # self.logger.debug('indx_vessel_cen: %s', indx_vessel_cen.shape)

        # ray_cell_x_ind, ray_cell_width = self.distrbute_ray_cells(ray_cell_x_ind_all)
        # self.logger.debug('ray_cell_x_ind: %s  %s', ray_cell_x_ind.shape, ray_cell_x_ind)
        # self.logger.debug('ray_cell_width:')

        # for i,width in enumerate(ray_cell_width):
        #     self.logger.debug('   %d %s', i+1, width)
        # # self.logger.debug('ray_cell_x_ind_all_update: %s  %s', ray_cell_x_ind_all_update.shape, ray_cell_x_ind_all_update)

        # vol_img_ref = np.full(self.params.size_im_enlarge, 255, dtype=float)
        # vol_img_ref = self.generate_small_fibers(ray_cell_x_ind, indx_skip_all, vol_img_ref)
        # vol_img_ref = self.generate_large_fibers(indx_ves_edges, indx_vessel_cen, vol_img_ref)

        # if self.params.is_exist_ray_cell:
        #     for idx, width in zip(ray_cell_x_ind, ray_cell_width):
        #         self.logger.debug('Generating ray cell: %s / %s', idx, width)
        #         vol_img_ref = self.generate_raycell(idx, width, vol_img_ref)

        # # Save the generated volume
        # self.save_slices(vol_img_ref, 'volImgBackBone')

        # # u1 and v1 are in a commented part of the code. Prob used in original code?
        # u, v, _, _ = self.generate_deformation(ray_cell_x_ind, indx_skip_all, indx_vessel_cen)
        # self.logger.debug('u.shape: %s  min/max: %s %s', u.shape, u.min(), u.max())
        # self.logger.debug('v.shape: %s  min/max: %s %s', v.shape, v.min(), v.max())

        # if self.params.is_exist_ray_cell:
        #     v_all_ray = self.ray_cell_shrinking(ray_cell_width, ray_cell_x_ind, v)
        #     v = v[..., np.newaxis] + v_all_ray
        #     self.logger.debug('vray   : %s  min/max: %s %s', v_all_ray.shape, v_all_ray.min(), v_all_ray.max())
        #     self.logger.debug('v.shape: %s  min/max: %s %s', v.shape, v.min(), v.max())

        # for i, slice_idx in enumerate(self.params.save_slice):
        #     self.logger.debug('Saving deformation for slice %d', slice_idx)
        #     if self.params.is_exist_ray_cell:
        #         v_slice = v[..., i]
        #     else:
        #         v_slice = v
        #     self.save_distortion(u, v_slice, slice_idx)

        #     img_interp = self.apply_deformation(vol_img_ref[..., slice_idx], u, v_slice)

        #     filename = os.path.join(self.root_dir, 'LocalDistVolume', f'volImgRef_{slice_idx+1:05d}.tiff')
        #     self.save_2d_img(img_interp, filename)
