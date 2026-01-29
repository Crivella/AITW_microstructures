"""Ray cells"""
import importlib
import logging
import os
import pathlib
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict

import nrrd
import numpy as np
import numpy.typing as npt
import requests
from PIL import Image
from scipy.interpolate import CubicSpline, RegularGridInterpolator, griddata

from . import distortion as dist
from . import ray_cells as rcl
from .clocks import Clock
from .fit_elipse import fit_elipse, fit_ellipse_6pt
from .loggers import add_file_logger, get_logger, set_console_level
from .params import BaseParams

# https://github.com/AI-TranspWood/AITW_microstructures/raw/refs/heads/main/wood_microstructure/BirchMicrostructure.pt
GIT_SOURCE = 'https://github.com'
GIT_OWNER = 'AI-TranspWood'
GIT_REPO = 'AITW_microstructures'
# GIT_REF = "refs/heads/main"
GIT_REF = '{commit}'
MODEL_URL_TEMPLATE = f'{GIT_SOURCE}/{GIT_OWNER}/{GIT_REPO}/raw/{GIT_REF}/wood_microstructure/{{model_name}}.pt'


class WoodMicrostructure(Clock, ABC):
    """Base class for wood microstructure generation"""
    ParamsClass: BaseParams = None

    model_commit: str = None

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
        """Prefix for saving files"""

    @property
    @abstractmethod
    def skip_cell_thick_rescale(self) -> int:
        """Rescale of ellipse point for fitting"""

    @abstractmethod
    def get_distortion_map(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Get initial distortion map"""
        pass

    @abstractmethod
    def get_grid_all(
        self, thick_all_valid_sub: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Get the location of grid nodes and the thickness (with random disturbance)"""
        pass

    @abstractmethod
    def generate_vessel_indexes(self, ray_cell_x_ind_all: npt.NDArray = None) -> npt.NDArray:
        """Generate the vessel indexes"""
        pass

    @abstractmethod
    def get_indx_skip_all(self, indx_vessel: npt.NDArray) -> npt.NDArray:
        """Get the indexes of the vessels to be skipped"""
        pass

    @abstractmethod
    def get_indx_ves_edges(self, indx_vessel: npt.NDArray) -> npt.NDArray:
        """Get the indexes of the vessels edges for ellipse fitting"""
        pass

    @abstractmethod
    def get_indx_vessel_cen(self, indx_vessel: npt.NDArray) -> npt.NDArray:
        """Get the indexes of the vessel centers"""
        pass

    def __init__(
            self,
            params: BaseParams, *args,
            outdir: str = None, show_img: bool = False,
            num_parallel = 1,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        self._slice_interest = None
        self.x_grid_all = None
        self.y_grid_all = None
        self.thickness_all_ray = None
        self.thickness_all_fiber = None

        self.show_img = show_img
        self.outdir = outdir or os.getenv('ROOT_DIR', '.')

        # Setup logging
        num = self.get_root_dir()
        log_file = os.path.join(self.root_dir, 'wood_microstructure.log')
        self.logger = get_logger(str(num))
        add_file_logger(self.logger, log_file)

        # Validate and set parameters
        if params.apply_global_deform and not params.all_slices:
            self.logger.warning('\n'.join([
                '',
                'Global deformation can only be applied when saving all slices.'
                'Setting `apply_global_deform` to False.',
                'Set `save_slice` to "all" to enable global deformation.',
            ]))
            params.apply_global_deform = False

        self.params = params

        save_param_file = os.path.join(self.root_dir, 'params.json')
        self.params.to_json(save_param_file)

        self.num_parallel = num_parallel
        self.logger.debug('num_parallel: %d', num_parallel)
        if num_parallel > 1:
            if self.params.surrogate:
                msg = 'Using batching with %d slices per inference'
            else:
                msg = 'Using multiprocessing with %d processes'
            self.logger.info(msg, num_parallel)
        else:
            self.logger.info('Running in single process mode')

        self.device = None
        self.surrogate = None
        if self.params.surrogate:
            self.load_surrogate_model()

    @property
    def weights_filename(self) -> str:
        """Get the name of the surrogate model"""
        cls_name = self.__class__.__name__
        return f'{cls_name}.pt'

    @property
    def weights_native_path(self) -> str:
        """Get the native path of the surrogate model"""
        dir_name = os.path.dirname(__file__)
        weight_file = os.path.join(dir_name, self.weights_filename)
        return weight_file

    @property
    def weights_home_path(self) -> str:
        """Get the home path of the surrogate model"""
        aitw_home = pathlib.Path.home() / '.aitw'
        model_dir = aitw_home / 'models'
        model_path = model_dir / self.weights_filename
        return model_path.as_posix()

    def load_surrogate_model(self):
        """Load the surrogate model"""
        cls_name = self.__class__.__name__
        try:
            self.torch = torch = importlib.import_module('torch')
        except ImportError as e:
            self.logger.error('Install the package with the \[surrogate] extra to use the surrogate model')
            sys.exit(1)
        from .surrogate import U_Net

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.surrogate = U_Net()
        self.surrogate.to(self.device)

        weight_file = self.weights_native_path
        try:
            self.surrogate.load_state_dict(torch.load(weight_file, map_location=self.device))
        except Exception as e:
            weight_file = self.weights_home_path
            try:
                pathlib.Path(weight_file).parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.error('Failed to create model directory for `%s`', cls_name)
                self.logger.error(str(e))
                sys.exit(1)

            if not os.path.exists(weight_file):
                try:
                    self.download_surrogate_model(weight_file)
                except Exception as e:
                    self.logger.error('Failed to download surrogate model weights for `%s`', cls_name)
                    self.logger.error(str(e))
                    sys.exit(1)
            try:
                self.surrogate.load_state_dict(torch.load(weight_file, map_location=self.device))
            except Exception as e:
                self.logger.error('Failed to load surrogate model weights from `%s`', weight_file)
                self.logger.error(str(e))
                sys.exit(1)

        self.logger.info('Surrogate model weights loaded from `%s`', weight_file)
        return weight_file

    def download_surrogate_model(self, save_path: str):
        """Download the surrogate model weights"""
        cls_name = self.__class__.__name__
        if self.model_commit is None:
            raise ValueError(f'Model cannot be downloaded for `{cls_name}`: `model_commit` is not set')
        url = MODEL_URL_TEMPLATE.format(model_name=cls_name, commit=self.model_commit)

        self.logger.info('Downloading surrogate model weights for `%s` from `%s`', cls_name, url)

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 4 * 1024  # 1 KB
        wrote = 0
        start_time = time.time()
        last_time = -5
        speed_units = ['B/s', 'KB/s', 'MB/s', 'GB/s']
        with open(save_path, 'wb') as f:
            for data in response.iter_content(block_size):
                wrote = wrote + len(data)
                f.write(data)
                elapsed_time = time.time() - start_time
                if elapsed_time - last_time > 1:
                    speed = wrote / elapsed_time
                    units = speed_units.copy()
                    unit = units.pop(0)
                    while speed > 1024 and len(units) > 1:
                        speed /= 1024
                        unit = units.pop(0)
                    last_time = elapsed_time
                    if total_size > 0:
                        percent = wrote * 100 / total_size
                        self.logger.debug(f'Downloading {save_path}: {percent:>4.2f}% at {speed:>6.2f} {unit}')
                    else:
                        self.logger.debug(f'Downloading {save_path}: {wrote:>10d} bytes at {speed:>6.2f} {unit}')

        if total_size != 0 and wrote != total_size:
            self.logger.error('Download failed: size mismatch')
            raise Exception('Download failed: size mismatch')

    def set_console_level(self, level: int):
        """Set the console logging level"""
        set_console_level(self.logger, level)

    def get_root_dir(self) -> int:
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
        return dir_cnt

    @property
    def slice_interest(self):
        """Get the slices of interest"""
        if self._slice_interest is None:
            gz = self.params.size_im_enlarge[2]
            ds = self.params.slice_interest_space

            slice_interest = np.arange(0, gz, ds)
            if slice_interest[-1] != gz:
                slice_interest = np.append(slice_interest, gz)
            self._slice_interest = slice_interest
        return self._slice_interest

    @Clock.register('ray_cell')
    @Clock.register('rcl:distribute')
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

    @abstractmethod
    def _get_small_fiber_exp(self, is_close_to_ray: npt.NDArray) -> npt.NDArray:
        """Get the exponent for small fiber generation"""
        pass

    @Clock.register('small_fibers')
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
        ray_cell_idx = np.unique(ray_cell_idx // 2)

        sie_x, sie_y, _ = self.params.size_im_enlarge
        gx, gy = self.params.x_grid.shape

        x_grid_all = self.x_grid_all
        y_grid_all = self.y_grid_all
        thick_all = self.thickness_all_fiber

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

        is_close_to_ray = np.zeros_like(t_all, dtype=bool)
        if ray_cell_idx.size:
            for j in range(1, ly - 1, 2):
                mm = min(np.min(np.abs(j - ray_cell_idx)), np.min(np.abs(j - ray_cell_idx - 1)))
                is_close_to_ray[:, j // 2] = mm <= 4
        exp_ellipse_2 = self._get_small_fiber_exp(is_close_to_ray)

        skip_cell_thick = 0  # TODO: Should this be a settable parameter?
        # for i_slice in range(sie_z):
        num_slices = len(self.params.save_slice)
        for slice_idx, i_slice in enumerate(self.params.save_slice):
            self.logger.debug('  Small fibers: idx=%d  %d/%d', i_slice, slice_idx + 1, num_slices)
            x_slice = x_grid_all[:,:, slice_idx]
            y_slice = y_grid_all[:,:, slice_idx]
            t_slice = thick_all[:,:, slice_idx]

            # Assignments are split into [:, 0::2] and [1::2, :] to keep into account staggering along x direction
            # every other row
            t_all[:, 0::2] = t_slice[1:-2:2, 1:-2:4]
            t_all[:, 1::2] = t_slice[2:-1:2, 3:-2:4]

            # Apply the same as above but for a generalized list of neighboring locations to get the points to fit
            # the ellipses with a shape of (lx, ly, 4, 2) ~ (GRID_X_IDX, GRID_Y_IDX, NEIGHBOR_IDX, X/Y)
            for i,(dx,dy) in enumerate(neigh_loc.T):
                slice_1_1 = slice(1+dx, -2+dx, 2)
                slice_2_1 = slice(2+dx, (-1+dx) or None, 2)
                slice_1_2 = slice(1+dy, -2+dy, 4)
                slice_2_2 = slice(3+dy, -2+dy, 4)
                for j,s in enumerate((x_slice, y_slice)):
                    point_coords[:, 0::2, i,j] = s[slice_1_1, slice_1_2]
                    point_coords[:, 1::2, i,j] = s[slice_2_1, slice_2_2]

            if skip_cell_thick == 0:
                point_coords[:,:, 1, 1] -= self.skip_cell_thick_rescale
                point_coords[:,:, 3, 1] += self.skip_cell_thick_rescale

            r1, r2, h, k = fit_elipse(point_coords)  # Estimate the coefficients of the ellipse. (lx, ly, 4)

            # Skip ellipse generation based on mask:
            # - fiber inside ray cell
            # - fiber inside a vessel
            h[overflow_mask] = -1234
            # Skip ellipse generation if fiber has ended along the Z axis
            h[self.get_fiber_end_condition(lx, ly, i_slice)] = -1234

            # The alternative is to write the full x/y grid and denote it into sub-domains based on the closest h/k
            # center and than use griddata to get the value of r1/r2/h/k on the full grid but this is slower
            for thick, _r1, _r2, _h, _k, exp in zip(
                t_all.flatten(), r1.flatten(), r2.flatten(), h.flatten(), k.flatten(), exp_ellipse_2.flatten()
            ):
                if _h == -1234:
                    # Skip ellipse generation
                    continue
                if np.any(np.isnan([_h, _k, _r1, _r2])):
                    self.logger.warning('NaN in ellipse parameters: h=%.1f k=%.1ff r1=%.1f r2=%.1f', _h, _k, _r1, _r2)
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
                    np.abs(rx_grid - _h)**exp / np.abs(_r1 - thick - skip_cell_thick)**exp +
                    np.abs(ry_grid - _k)**exp / np.abs(_r2 - thick - skip_cell_thick)**exp
                )

                vol_img_ref[rx_grid, ry_grid, slice_idx] /= 1 + np.exp(-(in_elipse_2 - 1) * 20)

        return vol_img_ref.astype(int)

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

    @Clock.register('large_fibers')
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

        sie_x, sie_y, _ = self.params.size_im_enlarge

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
                for slice_idx, i_slice in enumerate(self.params.save_slice):
                    point_coord = np.column_stack((
                        x_grid_all[six_pt_x, six_pt_y, slice_idx],
                        y_grid_all[six_pt_x, six_pt_y, slice_idx]
                    ))
                    r1, r2, h, k = fit_ellipse_6pt(point_coord)  # Estimate the coefficients of the ellipse.

                    thick = self.thickness_all_fiber[i1, j, slice_idx] + vessel_thicker
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
                    vol_img_ref[x_grid[cond], y_grid[cond], slice_idx] = 255 / mul[cond]

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

        vessel_end_loc = -np.random.rand(*shape, 1) * ray_cell_length
        while np.any(vessel_end_loc[...,-1] < lim):
            tmp = ray_cell_length + ray_cell_variance * np.random.randn(*shape, 1)
            tmp[tmp < rcl_d3] = rcl_t2
            tmp[tmp > rcl_t2] = rcl_t2
            vessel_end_loc = np.concatenate((vessel_end_loc, vessel_end_loc[..., -1:] + tmp), axis=-1)

        # Final filtering based on sie_x + ray_cell_length / 2 is done after to avoid altering the shape
        return vessel_end_loc.astype(int)

    @abstractmethod
    def _generate_raycell_cell_r(self, interp1: npt.NDArray, interp2: npt.NDArray, dx: npt.NDArray, k: int):
        """Get the value of `cell_r` for `generate_raycell`"""
        pass

    @abstractmethod
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
        pass

    @Clock.register('ray_cell')
    def generate_raycell(
            self, ray_idx: int, ray_width: npt.NDArray, input_volume: npt.NDArray,
            thickness_all: npt.NDArray
        ) -> npt.NDArray:
        """Generate ray cell

        Args:
            ray_idx (int): y-index of the ray cell
            ray_width (npt.NDArray): Ray cell width
            input_volume (npt.NDArray): Input 3D gray-scale image volume to modify
            thickness_all (npt.NDArray): Thickness of the ray cells

        Returns:
            npt.NDArray: Modified 3D gray-scale image volume with ray cells
        """
        vol_img_ref_final = np.copy(input_volume)

        slice_map = self.params.save_slice_map

        ray_cell_length = self.params.ray_cell_length
        ray_height = self.params.ray_height

        rcl_d2 = ray_cell_length / 2
        rcl_d2_r = np.round(ray_cell_length / 2)

        sie_x, _, sie_z = self.params.size_im_enlarge

        x_grid_all = self.x_grid_all
        y_grid_all = self.y_grid_all

        cell_end_thick = self.params.cell_end_thick

        ray_width = np.array(ray_width).astype(int)

        dx = np.arange(sie_x)

        # Generates 1 column_idx at a time with X and X+1
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
                t_idx = slice_map[t]

                vel = vessel_end_loc[i]
                vel = vel[vel <= sie_x + rcl_d2]

                interp_x0 = x_grid_all[:, column_idx, t_idx]
                interp_y0 = y_grid_all[:, column_idx, t_idx]
                interp_x1 = x_grid_all[:, column_idx + 1, t_idx]
                interp_y1 = y_grid_all[:, column_idx + 1, t_idx]
                interp_thick = thickness_all[:, column_idx, t_idx]
                tmp_2 = rcl_d2_r if m2 % 2 else 0

                try:
                    y_interp1_c = CubicSpline(interp_x0, interp_y0)(dx) - 1.5
                    y_interp2_c = CubicSpline(interp_x1, interp_y1)(dx) + 1.5
                    thick_interp_c = CubicSpline(interp_x0, interp_thick)(dx)
                except ValueError:
                    self.logger.warning('    WARNING: Spline interpolation failed')
                    continue

                cell_center = np.column_stack((
                    dx,
                    np.round((y_interp2_c + y_interp1_c)) / 2,
                    np.full(dx.shape, tmp_1)
                ))
                cell_r = self._generate_raycell_cell_r(y_interp1_c, y_interp2_c, dx, k)

                flag = -1
                check = len(vel) - 2
                for cnt,(vel_r, vel_r1) in enumerate(zip(vel[:-1], vel[1:])):
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

                    valid_idx = self._generate_raycell_valid_idx(vel_col_r, vel_col_r1, flag, cell_end_thick)
                    flag = 0 if cnt < check else 1

                    for idx in range(vel_col_r, vel_col_r1):
                        ##################
                        # TODO: This needs to be checked:
                        #  This commented part is carving out parallelipipedal boxes in the volume resetting it to
                        #  255 (filled). The area with the ray cells is re-emptied in the part after.
                        #  This is cause artefacts where the areas overlap with neighboring ray cells and vessels
                        #  causing sharp edges.
                        #  The only difference here between the original code is that we are doing an Z->Y->X
                        #  loop instaed of a Y->X->Z loop. This should change the order of the overlap but it should
                        #  still be there.
                        #  Here by just not doing it it seems to be working as we are already skipping the fiber and
                        #  vessel generation inside the ray cells.
                        ##################
                        # if j_slice == np.min(ray_width):
                        #     y_idxs = np.arange(int(y_interp1_c[idx]), int(y_interp2_c[idx] + 1))
                        #     z_idxs = np.arange(int(cell_center[idx, 2]), int(cell_neigh_pt[2, 1] + 1))
                        # elif j_slice == np.max(ray_width):
                        #     y_idxs = np.arange(int(y_interp1_c[idx]), int(y_interp2_c[idx] + 1))
                        #     z_idxs = np.arange(int(cell_neigh_pt[2, 0]), int(cell_center[idx, 2] + 1))
                        # else:
                        #     y_idxs = np.arange(int(y_interp1_c[idx]), int(y_interp2_c[idx] + 1))
                        #     z_idxs = np.arange(int(cell_neigh_pt[2, 0]), int(cell_neigh_pt[2, 1]))
                        # z_idxs = [slice_map[z] for z in z_idxs if z in slice_map]
                        # if z_idxs:
                        #     y,z = np.meshgrid(y_idxs, z_idxs)
                        #     vol_img_ref_final[idx, y, z] = 255

                        if idx not in valid_idx:
                            continue

                        for s in range(int(cell_neigh_pt[2, 0]), int(cell_neigh_pt[2, 1]) + 1):
                            if s not in slice_map:
                                continue
                            for j in range(int(cell_neigh_pt[1, 0]), int(cell_neigh_pt[1, 1]) + 1):
                                outer_elipse = (
                                    (j - cell_center[idx, 1])**2 / cell_r[idx, 0]**2 +
                                    (s - cell_center[idx, 2])**2 / cell_r[idx, 1]**2
                                )

                                if outer_elipse < 1:
                                    inner_elipse = (
                                        (j - cell_center[idx, 1])**2 / (cell_r[idx, 0] - thick_interp_c[idx])**2 +
                                        (s - cell_center[idx, 2])**2 / (cell_r[idx, 1] - thick_interp_c[idx])**2
                                    )
                                    vol_img_ref_final[idx, j, slice_map[s]] = int(
                                        (1 / (1 + np.exp(-(inner_elipse - 1) / .05))) * 255
                                    )

        return vol_img_ref_final

    @abstractmethod
    def _get_k_grid1(self, is_ctr: npt.NDArray, is_ctr_far: npt.NDArray, vess_cond: npt.NDArray) -> npt.NDArray:
        """Get the k_grid of parameters for the deformation map"""
        pass

    @abstractmethod
    def _get_k_grid2(
        self, k_grid: npt.NDArray, is_ctr: npt.NDArray, is_ctr_far: npt.NDArray, vess_cond: npt.NDArray
    ) -> npt.NDArray:
        """Regenerate part of the k_grid for different random numbers between U and V computation."""
        pass

    @abstractmethod
    def _get_sign_grid(self, vess_cond: npt.NDArray) -> npt.NDArray:
        """Get the sign grid of parameters for accumulating the deformation map"""
        pass

    @abstractmethod
    def _get_u1_v1(
        self, xc_grid: npt.NDArray, yc_grid: npt.NDArray,
        is_close_to_ray_far: npt.NDArray, sie_x: int, sie_y: int
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Get the large scale deformation u1 and v1"""
        pass

    @Clock.register('deformation')
    @Clock.register('deform:generate')
    def generate_deformation(
            self, ray_cell_idx: npt.NDArray, indx_skip_all: npt.NDArray, idx_vessel_cen: npt.NDArray
        ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Add complicated deformation to the volume image. The deformation fields are generated separately.
        Then, they are summed together. Here u, v are initialized to be zero. Then they are summed."""
        self.logger.info('=' * 80)
        self.logger.info('Generating deformation...')
        self.logger.debug('  ray_cell_idx: %s', ray_cell_idx.shape)
        sie_x, sie_y, _ = self.params.size_im_enlarge

        gx, gy = self.params.x_grid.shape

        u = np.zeros((sie_x, sie_y), dtype=float)
        v = np.zeros_like(u, dtype=float)

        lx = (gx - 1) // 2
        ly = (gy - 1) // 2

        x_slice = self.x_grid_all[:, :, 0]
        y_slice = self.y_grid_all[:, :, 0]

        skip_idx = set()
        # indx_skip_all: (num_vessels, 6, 2)
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

        # s_grid = np.sign(np.random.randn(lx, ly))
        # s_grid[cond] = -1

        s_grid = self._get_sign_grid(cond)
        k_grid = self._get_k_grid1(is_close_to_ray, is_close_to_ray_far, cond)

        for xc, yc, k, s in zip(xc_grid.flatten(), yc_grid.flatten(), k_grid.reshape(-1, 4), s_grid.flatten()):
            if (xc, yc) in skip_idx:
                continue
            xp, yp = dist.get_distortion_grid(xc, yc, sie_x, sie_y, self.local_distortion_cutoff)
            local_dist = dist.local_distort(xp, yp, xc, yc, k)
            # plt.contourf(xp, yp, local_dist, 50)
            # plt.colorbar()
            # plt.title(f'local_distort: {xc}, {yc}')
            # plt.tight_layout()
            # plt.show()
            u[xp, yp] += -s * local_dist

        k_grid = self._get_k_grid2(k_grid, is_close_to_ray, is_close_to_ray_far, cond)

        for xc, yc, k, s in zip(xc_grid.flatten(), yc_grid.flatten(), k_grid.reshape(-1, 4), s_grid.flatten()):
            if (xc, yc) in skip_idx:
                continue
            xp, yp = dist.get_distortion_grid(xc, yc, sie_x, sie_y, self.local_distortion_cutoff)
            local_dist = dist.local_distort(yp, xp, yc, xc, k)
            v[xp, yp] += -s * local_dist

        u1, v1 = self._get_u1_v1(xc_grid, yc_grid, is_close_to_ray_far, sie_x, sie_y)

        return u, v, u1, v1

    @Clock.register('deformation')
    @Clock.register('deform:rc_shrink')
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

        cnt = defaultdict(int)
        for idx in idx_all.flatten():
            cnt[int(idx)] += 1

        dx = np.arange(sie_x, dtype=int)
        _, y_grid = np.mgrid[0:sie_x, 0:sie_y]

        v_all = np.zeros((sie_x, sie_y, len(self.params.save_slice)), dtype=float)
        for i, slice_idx in enumerate(self.params.save_slice):
            self.logger.debug('slice: %d/%d', i, len(self.params.save_slice))
            x_node_grid = x_grid_all[..., i]
            y_node_grid = y_grid_all[..., i]
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
                    try:
                        y_node_grid_1 = CubicSpline(x_node_grid[:, idx], y_node_grid[:, idx])(dx)
                        y_node_grid_2 = CubicSpline(x_node_grid[:, idx + 2], y_node_grid[:, idx + 2])(dx)
                    except ValueError:
                        self.logger.warning('    WARNING: Spline interpolation failed')
                        continue

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

                # This is tied to the previous todo with the try/except on the splines
                if v1_all is None or v2_all is None:
                    self.logger.debug('    No valid deformation for this ray cell')
                    continue

                base_k += cnt[key]
                v_all[:, :, i] += coeff1[slice_idx] * v1_all + coeff2[slice_idx] * v2_all

        return v_all

    @Clock.register('deformation')
    @Clock.register('deform:local')
    def apply_local_deformation(
            self, vol_img_ref: npt.NDArray, u: npt.NDArray, v: npt.NDArray
        ) -> npt.NDArray:
        """Apply local deformation to the volume image"""
        self.logger.info('=' * 80)
        self.logger.info('Local deformation...')

        if self.surrogate is None or self.device is None:
            self._apply_local_deformation(vol_img_ref, u, v)
        else:
            self._apply_local_deformation_surrogate(vol_img_ref, u, v)

        return vol_img_ref

    def _apply_local_deformation(self, vol_img_ref: npt.NDArray, u: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
        """Apply the deformation to the volume image"""
        sie_x, sie_y, _ = self.params.size_im_enlarge
        x_grid, y_grid = np.mgrid[0:sie_x, 0:sie_y]
        x_interp = x_grid + u

        def _deform_slice(array_idx: int, grid_idx: int = None):
            gird_idx = array_idx if grid_idx is None else grid_idx
            self.logger.info('Applying distortion for slice %d', gird_idx)
            v_slice = v[..., array_idx] if self.params.is_exist_ray_cell else v
            y_interp = y_grid + v_slice
            Vq = griddata(
                (x_interp.flatten(), y_interp.flatten()),
                vol_img_ref[..., array_idx].flatten(),
                (x_grid, y_grid),
                method='linear',
                fill_value=255
            )
            img_interp = Vq.reshape(x_interp.shape)
            img_interp = np.clip(img_interp, 0, 255).astype(np.uint8)
            vol_img_ref[..., array_idx] = img_interp

        if self.num_parallel > 1:
            indexes = list(enumerate(self.params.save_slice))
            threads = []
            while indexes or threads:
                while len(threads) < self.num_parallel and indexes:
                    arr_idx, grid_idx = indexes.pop(0)
                    thread = threading.Thread(target=_deform_slice, args=(arr_idx, grid_idx))
                    thread.start()
                    threads.append(thread)
                torm = [i for i,t in enumerate(threads) if not t.is_alive()][::-1]
                for i in torm:
                    threads.pop(i)
                time.sleep(0.1)
        else:
            for arr_idx, grid_idx in enumerate(self.params.save_slice):
                _deform_slice(arr_idx, grid_idx)

        return vol_img_ref

    def _apply_local_deformation_surrogate(
            self, vol_img_ref: npt.NDArray, u: npt.NDArray, v: npt.NDArray
        ) -> npt.NDArray:
        """Apply the deformation using the surrogate model"""

        if self.num_parallel == 1:
            for i, slice_idx in enumerate(self.params.save_slice):
                self.logger.info('[SURROGATE] Applying distortion for slice %d', slice_idx)
                if self.params.is_exist_ray_cell:
                    v_slice = v[..., i]
                else:
                    v_slice = v

                img_interp = self.surrogate(
                    self.torch.from_numpy(vol_img_ref[..., i] / 255.0).float().unsqueeze(0).unsqueeze(0).to(self.device),
                    self.torch.from_numpy(u).float().unsqueeze(0).unsqueeze(0).to(self.device),
                    self.torch.from_numpy(v_slice).float().unsqueeze(0).unsqueeze(0).to(self.device)
                )
                img_interp = 255.0 * img_interp.squeeze().detach().cpu().numpy()

                vol_img_ref[..., i] = img_interp
        else:
            last = len(self.params.save_slice)
            chunk_edges = list(range(0, last + 1, self.num_parallel))
            if chunk_edges[-1] != last:
                chunk_edges.append(last)
            for start, end in zip(chunk_edges[:-1], chunk_edges[1:]):
                # TODO: Change shape of array in tool so that Z is the first dimension
                self.logger.info(
                    '[SURROGATE] Applying distortion for slices %d to %d', start, end - 1
                )
                u_t = self.torch.from_numpy(
                    np.full((end - start, *u.shape), u)
                ).float().unsqueeze(1).to(self.device)

                if self.params.is_exist_ray_cell:
                    v_t = self.torch.from_numpy(
                        np.transpose(v[..., start:end], axes=(2, 0, 1))
                    ).float().unsqueeze(1).to(self.device)
                else:
                    v_t = self.torch.from_numpy(
                        np.full((end - start, *v.shape), v)
                    ).float().unsqueeze(1).to(self.device)

                img_t = self.torch.from_numpy(
                    np.transpose(vol_img_ref[..., start:end], axes=(2, 0, 1)) / 255.0
                ).float().unsqueeze(1).to(self.device)

                # print(f'{img_t.shape = }, {u_t.shape = }, {v_t.shape = }')
                img_interp = self.surrogate(img_t, u_t, v_t)
                img_interp = 255.0 * img_interp.squeeze().detach().cpu().numpy()

                # print(f'{img_interp.shape = }')

                vol_img_ref[..., start:end] = np.transpose(img_interp, axes=(1, 2, 0))


        return img_interp

    @abstractmethod
    def _get_global_interp_grid(
            self,
            x_grid: npt.NDArray, y_grid: npt.NDArray, z_grid: npt.NDArray,
            u1: npt.NDArray, v1: npt.NDArray
        ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Get the interpolation grid for global deformation"""
        pass

    @Clock.register('deformation')
    @Clock.register('deform:global')
    def apply_global_deformation(self, vol_img_ref: npt.NDArray, u1: npt.NDArray, v1: npt.NDArray) -> npt.NDArray:
        """Apply global deformation to the volume image"""
        if not self.params.all_slices:
            raise RuntimeError('Global deformation is only applied when all slices are saved.')
        self.logger.info('=' * 80)
        self.logger.info('Global deformation...')

        sie_x, sie_y, _ = self.params.size_im_enlarge

        x_lin = np.arange(sie_x)
        y_lin = np.arange(sie_y)

        # self.logger.info(f'{sie_x = }, {sie_y = }, {sie_z = }')
        # self.logger.info(f'slice_interest: {self.slice_interest}')
        for slice_start, slice_end in zip(self.slice_interest[:-1], self.slice_interest[1:]):
            self.logger.debug(f'Global distortion slice {slice_start} to {slice_end}...')

            x_grid, y_grid, z_grid = np.mgrid[0:sie_x, 0:sie_y, slice_start:slice_end]

            x_interp, y_interp, z_interp, u_all_z, v_all_z = self._get_global_interp_grid(
                x_grid, y_grid, z_grid, u1, v1
            )

            if self.params.save_global_dist:
                for slice_idx in range(slice_start, slice_end):
                    self.save_global_distortion(
                        u_all_z[..., slice_idx - slice_start],
                        v_all_z[..., slice_idx - slice_start],
                        slice_idx
                    )

            self.logger.info(f'Interpolating... {x_grid.shape}')
            interp = RegularGridInterpolator(
                (x_lin, y_lin, np.arange(slice_start, slice_end)),
                vol_img_ref[..., slice_start:slice_end],
                method='linear',
                bounds_error=False,
                fill_value=255
            )
            vol_img_ref[..., slice_start:slice_end] = interp(
                np.stack((x_interp, y_interp, z_interp), axis=-1)
            ).astype(np.uint8)

            dirname = 'GlobalDistVolume'
            for slice_idx in range(slice_start, slice_end):
                filename = os.path.join(self.root_dir, dirname, f'volImgRef_{slice_idx+1:05d}.tiff')
                self.save_2d_img(vol_img_ref[..., slice_idx], filename)

        extra_size = np.array(self.params.extra_size, dtype=int)
        extra_sx_mid, extra_sy_mid, extra_sz_mid = extra_size // 2 + extra_size % 2

        vol_sx, vol_sy, vol_sz = self.params.size_volume

        final_volume = vol_img_ref[
            extra_sx_mid:extra_sx_mid + vol_sx,
            extra_sy_mid:extra_sy_mid + vol_sy,
            extra_sz_mid:extra_sz_mid + vol_sz
        ]

        dirname = 'FinalVolumeSlice'
        for idx in range(final_volume.shape[2]):
            filename = os.path.join(self.root_dir, dirname, f'volImgRef_{idx + 1:05d}.tiff')
            self.save_2d_img(final_volume[:,:,idx], filename)

        return final_volume

    @Clock.register('I/O')
    def create_dirs(self):
        """Ensure the output directories are created"""
        for dir_name in ['volImgBackBone', 'LocalDistVolume', 'LocalDistVolumeDispU', 'LocalDistVolumeDispV']:
            os.makedirs(os.path.join(self.root_dir, dir_name), exist_ok=True)

    def save_slices(self, vol_img_ref: npt.NDArray, dirname: str):
        """Save the requested slice of the generated volume image"""
        # self.logger.debug('vol_img_ref.shape: %s', vol_img_ref.shape)
        # self.logger.debug('min/max: %f %f', np.min(vol_img_ref), np.max(vol_img_ref))
        for i,slice_idx in enumerate(self.params.save_slice):
            filename = os.path.join(self.root_dir, dirname, f'volImgRef_{slice_idx+1:05d}.tiff')

            self.logger.debug('Saving slice %d to %s', slice_idx, filename)

            self.save_2d_img(vol_img_ref[:, :, i], filename, self.show_img)

    @staticmethod
    def ensure_dir(filename: str):
        """Ensure the directory exists"""
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)

    @staticmethod
    @Clock.register('I/O')
    @Clock.register('I/O:image')
    def save_2d_img(data: npt.NDArray, filename: str, show: bool = False):
        """Save 2D data to a TIFF file"""
        WoodMicrostructure.ensure_dir(filename)

        data[np.isnan(data)] = 255
        img = Image.fromarray(data.astype(np.uint8), mode='L')
        if show:
            img.show()
        img.save(filename)

    @staticmethod
    @Clock.register('I/O')
    @Clock.register('I/O:image')
    def save_3d_img(data: npt.NDArray, filename: str):
        """Save 3D data to a npy file"""
        WoodMicrostructure.ensure_dir(filename)
        data[np.isnan(data)] = 255

        _, ext = os.path.splitext(os.path.basename(filename))
        ext = ext.lower()

        if ext == '.nrrd':
            nrrd.write(filename, data.astype(np.uint8), index_order='C')
        elif ext == '.npy':
            np.save(filename, data)
        else:
            raise ValueError(f'Unsupported 3D image format: {ext}')

    @Clock.register('I/O')
    @Clock.register('I/O:csv')
    def save_local_distortion(self, u: npt.NDArray, v: npt.NDArray, slice_idx: int):
        """Save the distortion fields"""
        if not self.params.save_local_dist:
            return
        u_name = os.path.join(self.root_dir, 'LocalDistVolumeDispU', f'u_volImgRef_{slice_idx+1:05d}.csv')
        v_name = os.path.join(self.root_dir, 'LocalDistVolumeDispV', f'v_volImgRef_{slice_idx+1:05d}.csv')
        self.ensure_dir(u_name)
        self.ensure_dir(v_name)
        np.savetxt(u_name, np.round(u, decimals=4), delimiter=',', fmt='%0.4f')
        np.savetxt(v_name, np.round(v, decimals=4), delimiter=',', fmt='%0.4f')

    @Clock.register('I/O')
    @Clock.register('I/O:csv')
    def save_global_distortion(self, u: npt.NDArray, v: npt.NDArray, slice_idx: int):
        """Save the distortion fields"""
        if not self.params.save_global_dist:
            return
        u_name = os.path.join(self.root_dir, 'GlobalDistVolumeDispU', f'u_volImgRef_{slice_idx+1:05d}.csv')
        v_name = os.path.join(self.root_dir, 'GlobalDistVolumeDispV', f'v_volImgRef_{slice_idx+1:05d}.csv')
        self.ensure_dir(u_name)
        self.ensure_dir(v_name)
        np.savetxt(u_name, np.round(u, decimals=4), delimiter=',', fmt='%0.4f')
        np.savetxt(v_name, np.round(v, decimals=4), delimiter=',', fmt='%0.4f')

    def _generate_pipeline(self):
        """Pipeline for the wood microstructure generation"""
        v_fmt = self.params.save_volume_format.lower()
        # TODO: Check random seed behavior with multiprocessing
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

        indx_skip_all = self.get_indx_skip_all(vessel_all)
        indx_ves_edges = self.get_indx_ves_edges(vessel_all)
        indx_vessel_cen = self.get_indx_vessel_cen(vessel_all)
        self.logger.debug('indx_skip_all: %s', indx_skip_all.shape)
        self.logger.debug('indx_vessel: %s', indx_ves_edges.shape)
        self.logger.debug('indx_vessel_cen: %s', indx_vessel_cen.shape)

        ray_cell_x_ind, ray_cell_width = self.distrbute_ray_cells(ray_cell_x_ind_all)
        self.logger.debug('ray_cell_x_ind: %s  %s', ray_cell_x_ind.shape, ray_cell_x_ind)
        self.logger.debug('ray_cell_width:')
        for i,width in enumerate(ray_cell_width):
            self.logger.debug('   %d %s', i+1, width)

        shape = list(self.params.size_im_enlarge)
        shape[2] = len(self.params.save_slice)
        vol_img_ref = np.full(shape, 255, dtype=float)
        vol_img_ref = self.generate_small_fibers(ray_cell_x_ind, indx_skip_all, vol_img_ref)
        vol_img_ref = self.generate_large_fibers(indx_ves_edges, indx_vessel_cen, vol_img_ref)

        if self.params.is_exist_ray_cell:
            self.logger.info('Generating ray cells...')
            for i,(idx, width) in enumerate(zip(ray_cell_x_ind, ray_cell_width)):
                self.logger.info(f'Generating ray cell: {idx =}, {width = }  ({i+1}/{len(ray_cell_x_ind)})')
                vol_img_ref = self.generate_raycell(idx, width, vol_img_ref, self.thickness_all_ray)

        if self.params.save_volume_as_3d:
            filename = os.path.join(self.root_dir, 'FinalVolume3D', f'BeforeLocalVolume.{v_fmt}')
            self.save_3d_img(vol_img_ref, filename)

        # Save the generated volume
        self.save_slices(vol_img_ref, 'volImgBackBone')

        u, v, u1, v1 = self.generate_deformation(ray_cell_x_ind, indx_skip_all, indx_vessel_cen)
        self.logger.debug('u.shape: %s  min/max: %s %s', u.shape, u.min(), u.max())
        self.logger.debug('v.shape: %s  min/max: %s %s', v.shape, v.min(), v.max())
        self.logger.debug('u1.shape: %s  min/max: %s %s', u1.shape, u1.min(), u1.max())
        self.logger.debug('v1.shape: %s  min/max: %s %s', v1.shape, v1.min(), v1.max())

        if self.params.is_exist_ray_cell:
            v_all_ray = self.ray_cell_shrinking(ray_cell_width, ray_cell_x_ind, v)
            v = v[..., np.newaxis] + v_all_ray
            self.logger.debug('vray   : %s  min/max: %s %s', v_all_ray.shape, v_all_ray.min(), v_all_ray.max())
            self.logger.debug('v.shape: %s  min/max: %s %s', v.shape, v.min(), v.max())

        if compress_all_valid_sub.size:
            self.logger.info('Applying compression distortion to simulate late/earyl wood...')
            u += compress_all_valid_sub.reshape(-1, 1)

        for i, slice_idx in enumerate(self.params.save_slice):
            self.logger.debug('Saving distortion for slice %d', slice_idx)
            if self.params.is_exist_ray_cell:
                v_slice = v[..., i]
            else:
                v_slice = v
            self.save_local_distortion(u, v_slice, slice_idx)

        self.apply_local_deformation(vol_img_ref, u, v)

        for i, slice_idx in enumerate(self.params.save_slice):
            filename = os.path.join(self.root_dir, 'LocalDistVolume', f'volImgRef_{slice_idx+1:05d}.tiff')
            self.save_2d_img(vol_img_ref[..., i], filename)

        v_fmt = self.params.save_volume_format.lower()
        if self.params.apply_global_deform:
            if self.params.save_volume_as_3d:
                filename = os.path.join(self.root_dir, 'FinalVolume3D', f'BeforeGlobalVolume.{v_fmt}')
                self.save_3d_img(vol_img_ref, filename)
            vol_img_ref = self.apply_global_deformation(vol_img_ref, u1, v1)

        if self.params.save_volume_as_3d:
            filename = os.path.join(self.root_dir, 'FinalVolume3D', f'FinalVolume.{v_fmt}')
            self.save_3d_img(vol_img_ref, filename)

    def report(self):
        """Final report for the generation"""
        self.logger.info(self.report_clocks())

    def generate(self):
        """Generate the volume image"""
        self.create_dirs()
        self._generate_pipeline()

        self.report()
        self.logger.info('======== DONE ========')

    @classmethod
    def run_from_dict(
            cls,
            data: dict, output_dir: str = None, loglevel: int = logging.DEBUG,
            num_parallel: int = 1
        ) -> None:
        """Run the generator from a dictionary of parameters"""
        params = cls.ParamsClass.from_dict(data)
        ms = cls(params, outdir=output_dir, num_parallel=num_parallel)
        ms.set_console_level(loglevel)
        ms.generate()
