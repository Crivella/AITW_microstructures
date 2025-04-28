"""Ray cells"""
import os

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from . import ray_cells as rcl
from . import vessels as ves
from .clocks import Clock
from .fit_elipse import fit_elipse
from .microstructure import WoodMicrostructure


class BirchMicrostructure(WoodMicrostructure):
    save_prefix = 'SaveBirch'
    local_distortion_cutoff = 200
    ray_height_mod = 6

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

    def get_ray_cell_indexes(self) -> npt.NDArray:
        """Get ray cell indexes"""
        ly = len(self.params.y_vector)
        ray_cell_x_ind_all = np.empty((1, 0))
        if self.params.is_exist_ray_cell:
            ray_cell_x_ind_all = rcl.get_x_indexes(ly, 10, self.params.ray_space, 10)
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

    @Clock('small_fibers')
    def generate_small_fibers(
            self,
            ray_cell_idx: npt.NDArray,
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
        self.logger.info('=' * 80)
        self.logger.info('Generating small fibers...')
        vol_img_ref = np.copy(input_volume)

        neigh_loc = self.params.neighbor_local
        # skip_fiber_column = np.array(skip_fiber_column).flatten().astype(int)
        # skip_fiber_column = np.unique((skip_fiber_column[(skip_fiber_column % 2) == 0]) // 2)
        ray_cell_idx = np.unique(ray_cell_idx // 2)

        sie_x, sie_y, _ = self.params.size_im_enlarge
        gx, gy = self.params.x_grid.shape

        x_grid_all = self.x_grid_all
        y_grid_all = self.y_grid_all
        thick_all = self.thickness_all

        lx = (gx - 2) // 2
        ly = (gy - 2) // 2

        overflow_mask = np.zeros((lx, ly), dtype=bool)
        overflow_mask[:, ray_cell_idx] = True
        for ix, iy in indx_skip_all.reshape(-1, 2):
            if iy % 2 == 0:
                continue
            if (iy + 1) % 4 == 0:
                ix -= 1
            if ix % 2 == 0:
                continue
            overflow_mask[ix // 2, iy // 2] = True

        point_coords = np.empty((lx, ly, 4, 2))
        t_all = np.empty((lx, ly))
        skip_cell_thick = 0  # TODO: Should this be a settable parameter?
        # for i_slice in range(sie_z):
        for i_slice in self.params.save_slice:
            self.logger.debug('  Small fibers: %d/%s', i_slice, self.params.save_slice)
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
            h[overflow_mask] = None
            h[self.get_fiber_end_condition(lx, ly, i_slice)] = None

            # The alternative is to write the full x/y grid and denote it into sub-domains based on the closest h/k
            # center and than use griddata to get the value of r1/r2/h/k on the full grid but this is slower
            for thick, _r1, _r2, _h, _k in zip(t_all.flatten(), r1.flatten(), r2.flatten(), h.flatten(), k.flatten()):
                if _h is None:
                    continue
                if np.any(np.isnan([_h, _k, _r1, _r2])):
                    self.logger.warning('NaN in ellipse parameters: %f %f %f %f', _h, _k, _r1, _r2)
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
