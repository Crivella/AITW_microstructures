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
            ray_cell_idx (npt.NDArray): indexes of columns where not to generate fibers
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
        ray_cell_idx = np.unique((ray_cell_idx[(ray_cell_idx % 2) == 0]) // 2)

        sie_x, sie_y, _ = self.params.size_im_enlarge
        gx, gy = self.params.x_grid.shape

        x_grid_all = self.x_grid_all
        y_grid_all = self.y_grid_all
        thick_all = self.thickness_all_fiber

        lx = (gx - 2) // 2
        ly = (gy - 2) // 2

        point_coords = np.empty((lx, ly, 4, 2))
        t_all = np.empty((lx, ly))

        # exp_ellipse_1 = 4  # External contour
        exp_ellipse_2 = 5 + np.round(np.random.rand(lx, ly) * 2)  # Internal contour
        is_close_to_ray = np.zeros_like(t_all, dtype=bool)

        if ray_cell_idx.size:
            for j in range(1, ly - 1, 2):
                mm = min(np.min(np.abs(j - ray_cell_idx)), np.min(np.abs(j - ray_cell_idx - 1)))
                is_close_to_ray[:, j // 2] = mm <= 4
        mask = np.random.rand(lx, ly) < 0.8
        exp_ellipse_2[~is_close_to_ray & mask] -= 2

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
                point_coords[:,:, 1, 1] -= 1.5
                point_coords[:,:, 3, 1] += 1.5

            r1, r2, h, k = fit_elipse(point_coords)  # Estimate the coefficients of the ellipse. (lx, ly, 4)

            # Set a very high value for h in nodes that should be ignored
            h[:, ray_cell_idx] = 80000
            # if len(skip_idx) > 0:
            #     h[skip_idx[:, 0], skip_idx[:, 1]] = 80000
            for ix, iy in indx_skip_all.reshape(-1, 2):
                if iy % 2 == 0:
                    continue
                if (iy + 1) % 4 == 0:
                    ix -= 1
                if ix % 2 == 0:
                    continue
                # print(' Skipping:', i_slice, ix // 2, iy // 2)
                h[ix // 2, iy // 2] = 80000
            h[self.get_fiber_end_condition(lx, ly, i_slice)] = 80000

            # The alternative is to write the full x/y grid and denote it into sub-domains based on the closest h/k
            # center and than use griddata to get the value of r1/r2/h/k on the full grid but this is slower
            for thick, _r1, _r2, _h, _k, exp in zip(t_all.flatten(), r1.flatten(), r2.flatten(), h.flatten(), k.flatten(), exp_ellipse_2.flatten()):
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
                    (rx_grid - _h)**exp / (_r1 - thick - skip_cell_thick)**exp +
                    (ry_grid - _k)**exp / (_r2 - thick - skip_cell_thick)**exp
                )

                vol_img_ref[rx_grid, ry_grid, i_slice] /= 1 + np.exp(-(in_elipse_2 - 1) * 20)

        return vol_img_ref.astype(int)

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
        self.logger.debug('PARAM: thickness_all_ray.shape: %s', self.thickness_all_fiber.shape)
        self.logger.debug('PARAM: thickness_all_fiber.shape: %s', self.thickness_all_fiber.shape)

        ray_cell_x_ind_all = self.get_ray_cell_indexes()
        self.logger.debug('ray_cell_x_ind_all.shape: %s', ray_cell_x_ind_all.shape)
        self.logger.debug('ray_cell_x_ind_all: %s', ray_cell_x_ind_all)

        vessel_all = self.generate_vessel_indexes(ray_cell_x_ind_all)
        self.logger.debug('vessel_all.shape: %s', vessel_all.shape)
        self.logger.debug('vessel_all: %s', vessel_all)

        # indx_skip_all = ves.get_grid_idx_in_vessel(vessel_all)
        # indx_ves_edges = ves.get_grid_idx_edges(vessel_all)
        # indx_vessel_cen = vessel_all
        # self.logger.debug('indx_skip_all: %s', indx_skip_all.shape)
        # self.logger.debug('indx_vessel: %s', indx_ves_edges.shape)
        # self.logger.debug('indx_vessel_cen: %s', indx_vessel_cen.shape)

        ray_cell_x_ind, ray_cell_width = self.distrbute_ray_cells(ray_cell_x_ind_all)
        self.logger.debug('ray_cell_x_ind: %s  %s', ray_cell_x_ind.shape, ray_cell_x_ind)
        self.logger.debug('ray_cell_width:')
        for i,width in enumerate(ray_cell_width):
            self.logger.debug('   %d %s', i+1, width)
        # self.logger.debug('ray_cell_x_ind_all_update: %s  %s', ray_cell_x_ind_all_update.shape, ray_cell_x_ind_all_update)

        vol_img_ref = np.full(self.params.size_im_enlarge, 255, dtype=float)
        vol_img_ref = self.generate_small_fibers(ray_cell_x_ind, np.empty((0,2)), vol_img_ref)
        # vol_img_ref = self.generate_large_fibers(indx_ves_edges, indx_vessel_cen, vol_img_ref)

        # if self.params.is_exist_ray_cell:
        #     for idx, width in zip(ray_cell_x_ind, ray_cell_width):
        #         self.logger.debug('Generating ray cell: %s / %s', idx, width)
        #         vol_img_ref = self.generate_raycell(idx, width, vol_img_ref)  # TODO: check difference in `rayCellGenerate` vs `rayCellGenerateSpruce`

        # Save the generated volume
        self.save_slices(vol_img_ref, 'volImgBackBone')

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
