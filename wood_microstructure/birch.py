"""Ray cells"""
import os

import numpy as np
from scipy.interpolate import CubicSpline

from . import vessels as ves
from .microstructure import WoodMicrostructure


class BirchMicrostructure(WoodMicrostructure):
    save_prefix = 'SaveBirch'
    local_distortion_cutoff = 200

    def __init__(self, *args, **kwargs):
        """Initialize the BirchMicrostructure class"""
        super().__init__(*args, **kwargs)

        self.thickness_all = None

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

    def _generate(self):
        """Generate ray cells"""
        np.random.seed(self.params.random_seed)

        self.get_grid_all()

        self.logger.debug('PARAM: size_im_enlarge: %s', self.params.size_im_enlarge)
        self.logger.debug('PARAM: x_vector.shape: %s', self.params.x_vector.shape)
        self.logger.debug('PARAM: y_vector.shape: %s', self.params.y_vector.shape)
        self.logger.debug('PARAM: x_grid_all.shape: %s', self.x_grid_all.shape)

        ray_cell_x_ind_all = self.get_ray_cell_indexes()
        self.logger.debug('ray_cell_x_ind_all.shape: %s', ray_cell_x_ind_all.shape)
        self.logger.debug('ray_cell_x_ind_all: %s', ray_cell_x_ind_all)

        vessel_all = self.generate_vessel_indexes(ray_cell_x_ind_all)
        self.logger.debug('vessel_all.shape: %s', vessel_all.shape)
        # self.logger.debug('vessel_all: %s', vessel_all)

        indx_skip_all = ves.get_grid_idx_in_vessel(vessel_all)
        indx_ves_edges = ves.get_grid_idx_edges(vessel_all)
        indx_vessel_cen = vessel_all
        self.logger.debug('indx_skip_all: %s', indx_skip_all.shape)
        self.logger.debug('indx_vessel: %s', indx_ves_edges.shape)
        self.logger.debug('indx_vessel_cen: %s', indx_vessel_cen.shape)

        ray_cell_x_ind, ray_cell_width = self.distrbute_ray_cells(ray_cell_x_ind_all)
        self.logger.debug('ray_cell_x_ind: %s  %s', ray_cell_x_ind.shape, ray_cell_x_ind)
        self.logger.debug('ray_cell_width:')

        for i,width in enumerate(ray_cell_width):
            self.logger.debug('   %d %s', i+1, width)
        # self.logger.debug('ray_cell_x_ind_all_update: %s  %s', ray_cell_x_ind_all_update.shape, ray_cell_x_ind_all_update)

        vol_img_ref = np.full(self.params.size_im_enlarge, 255, dtype=float)
        vol_img_ref = self.generate_small_fibers(ray_cell_x_ind, indx_skip_all, vol_img_ref)
        vol_img_ref = self.generate_large_fibers(indx_ves_edges, indx_vessel_cen, vol_img_ref)

        if self.params.is_exist_ray_cell:
            for idx, width in zip(ray_cell_x_ind, ray_cell_width):
                self.logger.debug('Generating ray cell: %s / %s', idx, width)
                vol_img_ref = self.generate_raycell(idx, width, vol_img_ref)

        # Save the generated volume
        self.create_dirs()
        self.save_slices(vol_img_ref, 'volImgBackBone')

        # u1 and v1 are in a commented part of the code. Prob used in original code?
        u, v, _, _ = self.generate_deformation(ray_cell_x_ind, indx_skip_all, indx_vessel_cen)
        self.logger.debug('u.shape: %s  min/max: %s %s', u.shape, u.min(), u.max())
        self.logger.debug('v.shape: %s  min/max: %s %s', v.shape, v.min(), v.max())

        if self.params.is_exist_ray_cell:
            v_all_ray = self.ray_cell_shrinking(ray_cell_width, ray_cell_x_ind, v)
            v = v[..., np.newaxis] + v_all_ray
            self.logger.debug('vray   : %s  min/max: %s %s', v_all_ray.shape, v_all_ray.min(), v_all_ray.max())
            self.logger.debug('v.shape: %s  min/max: %s %s', v.shape, v.min(), v.max())

        for i, slice_idx in enumerate(self.params.save_slice):
            self.logger.debug('Saving deformation for slice %d', slice_idx)
            if self.params.is_exist_ray_cell:
                v_slice = v[..., i]
            else:
                v_slice = v
            self.save_distortion(u, v_slice, slice_idx)

            img_interp = self.apply_deformation(vol_img_ref[..., slice_idx], u, v_slice)

            filename = os.path.join(self.root_dir, 'LocalDistVolume', f'volImgRef_{slice_idx+1:05d}.tiff')
            self.save_2d_img(img_interp, filename)
