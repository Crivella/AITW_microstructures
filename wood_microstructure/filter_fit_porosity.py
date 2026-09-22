"""
filter_fit_porosity.py — Geometry filter for OpenLB interpolated boundary conditions.

Features:
  - SDF-based porosity control: --porosity with --sdf-sigma for "acid treatment" simulation
  - Grayscale-aware processing: treats raw intensity as continuous density field
  - Downsampling and smoothing: efficient multi-resolution processing
  - Supersampling: high-quality upsampling via anti-aliasing
  - Padding/cropping: add pore space or crop geometry
  - Validation: checks for isolated voxels and geometry quality

Binary convention: 0 = pore, 255 = solid
"""
import json
import os
import sys

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from . import myio, utils
from .clocks import Clock
from .params import FitPorosityParams
from .pipeline import Pipeline


class FitPorosity(Pipeline[FitPorosityParams]):
    ParamsClass = FitPorosityParams
    save_prefix = 'fit_porosity'
    logname = 'fit_porosity'

    def init_params(self, params: FitPorosityParams):
        """Initialize parameters"""
        super().init_params(params)

        self.data = myio.read_volume(self.params.input_file).astype(np.float32)
        self.geom_slice = (slice(None), slice(None), slice(None))

    def init_pipeline(self):
        """Initialize pipeline"""
        tasks = self.tasks
        tasks.append((self.preproces_image, [], {}, False))
        tasks.append((self.calc_original_porosity, [], {}, False))
        tasks.append((self.downsample, [], {}, self.params.down > 1))
        tasks.append((self.lowres_smooth, [], {}, self.params.smooth_low_iters > 0))
        tasks.append((self.thicken, [], {}, self.params.thicken > 0))
        tasks.append((self.upsample, [], {}, self.params.down > 1))
        tasks.append((self.supersample, [], {}, self.params.upsample_intermediate is not None))
        tasks.append((self.final_smooth, [], {}, self.params.final_smooth_iters > 0))
        tasks.append((self.remove_isolated_voxels, [], {}, True))
        tasks.append((self.majority_filter, [], {}, self.params.majority_filter > 0))
        tasks.append((self.add_padding, [], {}, self.params.pad != 0))
        tasks.append((
            self.adjust_porosity_post, [], {},
            self.params.porosity is not None and self.params.adjust_porosity_post
        ))
        tasks.append((self.validate, [], {}, True))
        tasks.append((self.save_results, [], {}, True))

    def _is_binary(self):
        """Check if the input data is in binary format or grayscale format."""
        if np.unique(self.data).size <= 2:
            return True

    def _compute_sdf_from_grayscale(self) -> npt.NDArray:
        """Compute SDF from grayscale image."""
        solid_high = self.params.solid_is_high

        data_min = float(np.min(self.data))
        data_max = float(np.max(self.data))

        if data_min == data_max:
            return np.ones_like(self.data, dtype=np.float32) * 100.0

        norm = (self.data - data_min) / (data_max - data_min)

        if solid_high:
            sdf = (0.5 - norm) * 40.0
        else:
            sdf = (norm - 0.5) * 40.0

        return sdf.astype(np.float32)

    @Clock.register(['adjust_porosity', 'sdf'])
    def adjust_porosity_sdf(self):
        """Adjust porosity by thresholding SDF of raw grayscale image."""
        self.logger.info(f"Adjusting porosity via SDF thresholding (grayscale input)...")
        target_porosity = self.params.porosity
        tolerance = 0.001

        sdf_sigma = self.params.sdf_sigma

        self.logger.info(f"Adjusting porosity via SDF threshold (acid treatment simulation)...")
        self.logger.info(f"  Target porosity: {target_porosity:.6f}")
        self.logger.info(f"  SDF blur sigma: {sdf_sigma} (boundary smearing)")

        sdf = self._compute_sdf_from_grayscale()
        sdf_smooth_arr = ndimage.gaussian_filter(sdf, sigma=sdf_sigma)

        # Determine SDF threshold range with binary search
        thr_min = float(np.min(sdf_smooth_arr))
        thr_max = float(np.max(sdf_smooth_arr))
        self.logger.info(f"  SDF range: [{thr_min:.2f}, {thr_max:.2f}]")
        self.logger.info(f"  Searching for optimal SDF threshold...")

        for iteration in range(20):
            thr_mid = (thr_min + thr_max) / 2.0

            porosity_test = np.mean(sdf_smooth_arr >= thr_mid)
            error = abs(porosity_test - target_porosity)

            self.logger.debug(
                f"  iter {iteration+1}: SDF_threshold={thr_mid:+.4f}, porosity={porosity_test:.6f}, error={error:.6f}"
            )

            if error < tolerance:
                self.logger.debug(f"  ✓ Converged!")
                break

            if porosity_test < target_porosity:
                # Porosity too low - need more pore, so need LOWER (more negative) threshold
                thr_max = thr_mid
            else:
                # Porosity too high - need less pore, so need HIGHER (more positive) threshold
                thr_min = thr_mid

            if abs(thr_max - thr_min) < 1e-6:
                self.logger.debug(f"    Converged (threshold range < 1e-6)")
                break

        res = (sdf_smooth_arr < thr_mid).astype(np.uint8)
        final_porosity = self._calculate_porosity(res)

        self.logger.info(f"  Erosion level (SDF threshold): {thr_mid:+.4f} voxel distance")
        self.logger.info(f"  Result porosity: {final_porosity:.6f}")
        self.logger.info(f"  Target: {target_porosity:.6f}, Error: {abs(final_porosity - target_porosity):.6f}")

        return res

    def preproces_image(self):
        """Preprocess the image"""
        if self._is_binary():
            if np.max(self.data) >= 254.99:
                self.logger.info(f"Detected binary input (0/255) — rescaling to 0/1 for processing.")
                self.data /= 255.0
        else:
            if self.params.porosity is not None:
                self.data = self.adjust_porosity_sdf()
            else:
                self.logger.info(f"Binarizing grayscale input using threshold: {self.params.threshold:.6f}")
                self.data = utils.binarize_volume(self.data, threshold=self.params.threshold) / 255.0
                if not self.params.solid_is_high:
                    self.data = 1.0 - self.data

        self.initial_shape = self.data.shape

    @staticmethod
    def _calculate_porosity(arr, geometry_slice: tuple[slice, slice, slice] = None) -> float:
        """Calculate the porosity of the image"""
        if geometry_slice is not None:
            arr_region = arr[geometry_slice]
        else:
            arr_region = arr

        porosity = np.mean(arr_region == 0)
        return porosity

    def calc_original_porosity(self):
        """Calculate the porosity of the image"""
        self.original_porosity = self._calculate_porosity(self.data)
        self.logger.info(f"Original porosity: {self.original_porosity:.6f}")

    def _block_majority_downsample(self, arr: npt.NDArray, factor: int) -> npt.NDArray:
        """Downsample the image using block majority voting"""
        if factor <= 1:
            return arr

        nx, ny, nz = arr.shape
        rx = (factor - nx % factor) % factor
        ry = (factor - ny % factor) % factor
        rz = (factor - nz % factor) % factor
        if rx or ry or rz:
            arr = np.pad(arr, ((0, rx), (0, ry), (0, rz)), constant_values=0)

        nx2, ny2, nz2 = arr.shape
        blk = arr.reshape(nx2 // factor, factor, ny2 // factor, factor, nz2 // factor, factor)
        solid_count = np.sum(blk, axis=(1, 3, 5))
        majority = solid_count > (factor**3 / 2)
        return (majority).astype(np.uint8)

    def downsample(self):
        """Downsample the image"""
        if self.params.down <= 1:
            self.logger.debug(f"No downsampling applied (down={self.params.down})")
            return
        self.data = self._block_majority_downsample(self.data, self.params.down)

        porosity = self._calculate_porosity(self.data)
        self.logger.debug(f"  After low-res smoothing: porosity={porosity:.6f}, shape={self.data.shape}")

    @staticmethod
    def _compute_sdf(arr: npt.NDArray) -> npt.NDArray:
        """Compute the signed distance function (SDF) of the image"""
        solid = arr > 0
        d_pore = ndimage.distance_transform_edt(~solid).astype(np.float32)
        d_solid = ndimage.distance_transform_edt(solid).astype(np.float32)

        sdf = d_pore
        sdf[solid] = -d_solid[solid]

        return sdf

    @staticmethod
    def _sdf_smooth(arr: npt.NDArray, sigma: float) -> npt.NDArray:
        """Smooth the image using SDF smoothing"""
        sdf = FitPorosity._compute_sdf(arr)
        sdf_g = ndimage.gaussian_filter(sdf, sigma=sigma)
        return (sdf_g < 0).astype(np.uint8)

    @Clock.register(['smooth', 'lowres'])
    def lowres_smooth(self):
        """Smooth the image at low resolution"""
        if self.params.smooth_low_iters <= 0:
            self.logger.debug(f"No low-resolution smoothing applied (smooth_low_iters={self.params.smooth_low_iters})")
            return
        for i in range(self.params.smooth_low_iters):
            self.data = self._sdf_smooth(self.data, self.params.smooth_sigma)

        porosity = self._calculate_porosity(self.data)
        self.logger.debug(f"  After low-res smoothing: porosity={porosity:.6f}")

    def _thicken_solid(self, arr: npt.NDArray, iterations: int) -> npt.NDArray:
        """Thicken the solid phase of the image"""
        struct = ndimage.generate_binary_structure(3, 1)
        solid = ndimage.binary_dilation(arr > 0, structure=struct, iterations=iterations)
        return solid.astype(np.uint8)

    def thicken(self):
        """Thicken the solid phase of the image"""
        if self.params.thicken <= 0:
            self.logger.debug(f"No thickening applied (thicken={self.params.thicken})")
            return
        self.data = self._thicken_solid(self.data, self.params.thicken)

        porosity = self._calculate_porosity(self.data)
        self.logger.debug(f"  After thickening: porosity={porosity:.6f}")

    def _upsample_nearest(
        self, arr: npt.NDArray, factor: int, target_shape: tuple[int, int, int] = None
    ) -> npt.NDArray:
        """Upsample the image using nearest-neighbor interpolation"""
        arr  = self.data
        arr = np.repeat(arr, factor, axis=0)
        arr = np.repeat(arr, factor, axis=1)
        arr = np.repeat(arr, factor, axis=2)

        if target_shape is not None:
            # Adjust to exact target shape if needed
            tx, ty, tz = target_shape
            arr = arr[:tx, :ty, :tz]
            pad = (
                (0, max(0, tx - arr.shape[0])),
                (0, max(0, ty - arr.shape[1])),
                (0, max(0, tz - arr.shape[2]))
            )
            if any(p[1] > 0 for p in pad):
                arr = np.pad(arr, pad, mode='constant', constant_values=0)
        return arr

    def upsample(self):
        """Upsample the image back to original resolution"""
        if self.params.down <= 1:
            self.logger.debug(f"No upsampling applied (down={self.params.down})")
            return
        self.data = self._upsample_nearest(self.data, self.params.down, target_shape=self.initial_shape)

    def supersample(self):
        """Supersample the image"""
        ui = self.params.upsample_intermediate
        uf = self.params.upsample_final
        if ui is None or uf is None:
            self.logger.debug(f"No supersampling applied")
            return
        self.logger.info(f"** SUPERSAMPLING **  High-quality upsampling via anti-aliasing:")
        self.logger.info(f"  Current shape: {self.data.shape}")
        self.logger.info(f"  Strategy: upsample {ui}× → downsample to {uf}× net")

        intermediate_factor = int(np.round(ui))
        self.logger.info(f"Step 1: Upsampling {intermediate_factor}× to intermediate resolution…")

        self.data = self._upsample_nearest(self.data, intermediate_factor)
        self.logger.info(f"  Intermediate shape: {self.data.shape}")
        self.logger.info(f"  Porosity: {self._calculate_porosity(self.data):.6f}")

        # Downsample to final resolution
        downsample_factor = ui / uf
        self.logger.info(f"Step 2: Downsampling by {downsample_factor:.2f}× (anti-aliasing)…")

        target_shape = tuple(int(np.round(d * uf)) for d in self.data.shape)
        if downsample_factor > 1.0:
            ds_factor = int(np.round(downsample_factor))
            if abs(downsample_factor - ds_factor) > 0.01:
                self.logger.warning(f"Non-integer downsample factor {downsample_factor:.2f} rounded to {ds_factor}")

            self.data = self._block_majority_downsample(self.data, ds_factor)

            # Adjust to exact target shape if needed
            if self.data.shape != target_shape:
                self.logger.info(f"Adjusting to exact target shape {target_shape}…")
                self.data = self._upsample_nearest(self.data, 1, target_shape=target_shape)
        else:
            self.logger.info(f"No downsampling needed (downsample_factor={downsample_factor:.2f})")

        self.logger.info(f"  Final shape: {self.data.shape}")
        self.logger.info(f"  Porosity after supersampling: {self._calculate_porosity(self.data):.6f}")
        self.logger.info(f"  Net upsampling: {self.data.shape[0]/self.data.shape[0]:.2f}×")

    @Clock.register(['smooth', 'final'])
    def final_smooth(self):
        """Smooth the image at full resolution"""
        iters = self.params.final_smooth_iters
        if iters <= 0:
            self.logger.debug(f"No final smoothing applied (final_smooth_iters={iters})")
            return

        sigma = self.params.final_smooth_sigma
        self.logger.info(f"** FINAL SMOOTHING (after upsampling) **")
        self.logger.info(f"SDF smoothing at full resolution ({iters} × σ={sigma})…")

        for i in range(iters):
            self.data = self._sdf_smooth(self.data, sigma)

        porosity = self._calculate_porosity(self.data)
        self.logger.debug(f"  After final smoothing: porosity={porosity:.6f}")

    @Clock.register(['find_isolated_kernel'])
    def _find_isolated_kernel(self, arr: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Detect isolated voxels using NumPy vectorization."""
        nx, ny, nz = arr.shape
        solid = arr == 1
        isolated_solid = np.zeros_like(solid)
        isolated_pore = np.zeros_like(solid)

        neigh_count = np.zeros((nx - 2, ny - 2, nz - 2), dtype=arr.dtype)  # Prepare result array excluding boundaries
        neigh_count = (
            (solid[:-2, 1:-1, 1:-1]) + (solid[2:, 1:-1, 1:-1]) +
            (solid[1:-1, :-2, 1:-1]) + (solid[1:-1, 2:, 1:-1]) +
            (solid[1:-1, 1:-1, :-2]) + (solid[1:-1, 1:-1, 2:])
        )

        isolated_solid[1:-1, 1:-1, 1:-1] = (solid[1:-1, 1:-1, 1:-1]) & (neigh_count == 0)
        isolated_pore[1:-1, 1:-1, 1:-1] = (~solid[1:-1, 1:-1, 1:-1]) & (neigh_count == 6)

        return isolated_solid, isolated_pore

    def remove_isolated_voxels(self):
        """Remove isolated voxels from the image"""
        total_solid = total_pore = 0
        for sweep in range(10):
            isolated_solid, isolated_pore = self._find_isolated_kernel(self.data)

            r_solid = np.sum(isolated_solid)
            r_pore = np.sum(isolated_pore)

            total_solid += r_solid
            total_pore += r_pore

            self.data[isolated_solid] = 0  # Convert isolated solid to pore
            self.data[isolated_pore] = 1

            if r_solid == 0 and r_pore == 0:
                self.logger.debug(f"No more isolated voxels found after {sweep} sweeps.")
                break

        self.logger.info(f"Removed {total_solid} isolated solid, {total_pore} isolated pore voxels")

    @Clock.register(['thin_protrusions_kernel'])
    def _remove_thin_protrusions_kernel(
            self,
            arr: npt.NDArray, neighbor_threshold: float, inplace=False
        ) -> npt.NDArray:
        nx, ny, nz = arr.shape
        res = arr if inplace else arr.copy()

        base = res[1:-1, 1:-1, 1:-1]  # Central part of the array

        cnt_phase = np.zeros((nx - 2, ny - 2, nz - 2), dtype=arr.dtype)  # Prepare result array excluding boundaries
        cnt_phase += (base == arr[:-2, 1:-1, 1:-1])   # left neighbor
        cnt_phase += (base == arr[2:, 1:-1, 1:-1])    # right neighbor
        cnt_phase += (base == arr[1:-1, :-2, 1:-1])   # front neighbor
        cnt_phase += (base == arr[1:-1, 2:, 1:-1])    # back neighbor
        cnt_phase += (base == arr[1:-1, 1:-1, :-2])   # bottom neighbor
        cnt_phase += (base == arr[1:-1, 1:-1, 2:])    # top neighbor

        cond = cnt_phase <= neighbor_threshold

        base[cond] = 1 - base[cond]  # Flip the phase for isolated voxels

        return res

    def majority_filter(self):
        """Apply majority filter to the image"""
        if self.params.majority_filter <= 0:
            self.logger.debug(f"No majority filtering applied (majority_filter={self.params.majority_filter})")
            return

        arr = self.data.copy()
        self.logger.info(f"Applying majority filter to remove surface irregularities...")
        for i in range(self.params.majority_filter):
            arr = self._remove_thin_protrusions_kernel(arr, neighbor_threshold=3, inplace=True)
            changed_voxels = np.sum(self.data != arr)
            self.logger.info(f"  Iteration {i+1}: changed {changed_voxels} voxels")

        changed_voxels = np.sum(self.data != arr)
        self.logger.info(f"  Majority filter changed {changed_voxels} voxels")

        self.data = arr

    def add_padding(self):
        """Add padding to the image"""
        thickness = self.params.pad
        xy_only = self.params.pad_xy_only

        if thickness == 0:
            self.logger.debug(f"No padding applied (pad={thickness})")
            self.geom_slice = (slice(None), slice(None), slice(None))
            return

        if thickness > 0:
            self.logger.info(f"Adding {thickness} voxels of pore padding (xy_only={xy_only})...")
            if xy_only:
                pad_width = ((thickness, thickness), (thickness, thickness), (0, 0))
            else:
                pad_width = ((thickness, thickness), (thickness, thickness), (thickness, thickness))

            self.data = np.pad(self.data, pad_width, mode='constant', constant_values=0)

            nx, ny, nz = self.data.shape
            if xy_only:
                self.geom_slice = (slice(thickness, nx - thickness), slice(thickness, ny - thickness), slice(None))
            else:
                self.geom_slice = (
                    slice(thickness, nx - thickness),
                    slice(thickness, ny - thickness),
                    slice(thickness, nz - thickness)
                )
        else:
            self.logger.info(f"Cropping {-thickness} voxels from edges (xy_only={xy_only})...")
            crop = -thickness
            nx, ny, nz = self.data.shape

            if xy_only:
                if crop * 2 >= nx or crop * 2 >= ny:
                    self.logger.error(f"Cropping too large for XY dimensions: crop={crop}, shape={self.data.shape}")
                    sys.exit(1)
                self.data = self.data[crop:nx - crop, crop:ny - crop, :]
            else:
                if crop * 2 >= nx or crop * 2 >= ny or crop * 2 >= nz:
                    self.logger.error(f"Cropping too large for dimensions: crop={crop}, shape={self.data.shape}")
                    sys.exit(1)
                self.data = self.data[crop:nx - crop, crop:ny - crop, crop:nz - crop]

            self.geom_slice = (slice(None), slice(None), slice(None))

    @Clock.register(['adjust_porosity', 'post'])
    def adjust_porosity_post(self):
        """Adjust the porosity of the image after processing"""
        if self.params.porosity is None or not self.params.adjust_porosity_post:
            self.logger.debug(f"No post-processing porosity adjustment applied.")
            return

        target = self.params.porosity
        current = self._calculate_porosity(self.data, geometry_slice=self.geom_slice)

        if abs(current - target) < 1e-5:
            self.logger.info(f"Porosity already within tolerance: {current:.6f} (target: {target:.6f})")
            return

        self.logger.info(f"Adjusting porosity post-processing: current={current:.6f}, target={target:.6f}")
        self.logger.info(f"  Using SDF threshold tuning (post-processing)...")

        sdf = self._compute_sdf(self.data)
        sdf_smooth = ndimage.gaussian_filter(sdf, sigma=0.5)

        thr_min = -5.0
        thr_max = 5.0

        best = 0.0
        best_error = abs(current - target)

        for iteration in range(20):
            thr_mid = (thr_min + thr_max) / 2.0
            adjusted = (sdf_smooth < thr_mid).astype(np.uint8)
            porosity_test = self._calculate_porosity(adjusted, geometry_slice=self.geom_slice)
            error = abs(porosity_test - target)

            self.logger.debug(
                f"  iter {iteration+1}: SDF_threshold={thr_mid:+.4f}, porosity={porosity_test:.6f}, error={error:.6f}"
            )

            if error < best_error:
                best_error = error
                best = thr_mid

            if error < 1e-5:
                self.logger.debug(f"  ✓ Converged!")
                break

            if porosity_test < target:
                thr_max = thr_mid
            else:
                thr_min = thr_mid

            if abs(thr_max - thr_min) < 1e-6:
                self.logger.debug(f"    Converged (threshold range < 1e-6)")
                break

        self.data = (sdf_smooth < best).astype(np.uint8)
        self.final_porosity = self._calculate_porosity(self.data, geometry_slice=self.geom_slice)

        self.logger.info(f"  Final SDF threshold: {best:+.4f}")
        self.logger.info(f"  Porosity adjusted: {current:.6f} → {self.final_porosity:.6f} (target: {target:.6f})")
        self.logger.info(f"  Error after adjustment: {abs(self.final_porosity - target):.6f}")

    @Clock.register(['validate'])
    def validate(self):
        """Validate the geometry of the image"""
        arr_region = self.data[self.geom_slice]

        solid = arr_region > 0

        kernel = np.array([
            [[0,0,0],[0,1,0],[0,0,0]],
            [[0,1,0],[1,0,1],[0,1,0]],
            [[0,0,0],[0,1,0],[0,0,0]]
        ], dtype=np.uint8)

        neighbour_count = ndimage.convolve(solid.astype(np.uint8), kernel, mode='nearest', cval=0)
        remaining_iso = int(np.sum(solid & (neighbour_count == 0)))

        self.logger.info(f"Validation results:")
        self.logger.info(f"  Remaining isolated solid voxels: {remaining_iso}")
        self.logger.info(f"  Porosity (geometry only): {self._calculate_porosity(arr_region):.6f}")

        if remaining_iso > 0:
            self.logger.warning(f"  {remaining_iso} Remaining isolated solid voxels detected.")
        else:
            self.logger.info(f"  [bold green]SUCCESS[/bold green]: geometry suitable for interpolated BC")

        self.remaining_isolated_voxels = remaining_iso

        target = self.params.porosity
        if target:
            current = self._calculate_porosity(arr_region)
            if abs(current - target) > 0.05:
                self.logger.warning(f"Final porosity differs from target: {current:.6f} (target: {target:.6f})")
                self.logger.warning(
                    f"This is expected when processing (smoothing, filtering) substantially changes geometry."
                )
                self.logger.warning(f"Try: adjust-porosity-post for fine-tuning after processing")

    def _save_image(self):
        """Save the output image"""
        path = os.path.join(self.root_dir, 'fit_porosity_volume.vti')
        self.logger.info(f"Saving output volume to: {path}  shape={self.data.shape}")
        myio.write_volume(path, self.data * 255)

    def _save_params(self):
        """Save the porosity information"""
        final_porosity = self._calculate_porosity(self.data, geometry_slice=self.geom_slice)
        dct = {
            'original_porosity': self.original_porosity,
            'final_porosity': final_porosity,
            'remaining_isolated_voxels': self.remaining_isolated_voxels,
        }

        dct['target_porosity'] = self.params.porosity

        fpath = os.path.join(self.root_dir, 'output_params.json')
        self.logger.info(f"Saving porosity information to: {fpath}")
        with open(fpath, 'w') as f:
            json.dump(dct, f, indent=4)

        self.logger.info('=' * 80)
        if self.params.porosity is not None:
            self.logger.info(f"Target porosity: {self.params.porosity:.6f}")
        self.logger.info(f"Original porosity: {self.original_porosity:.6f}")
        self.logger.info(f"Final porosity: {final_porosity:.6f}")
        self.logger.info(f"Remaining isolated solid voxels: {self.remaining_isolated_voxels}")
        self.logger.info(f"Output volume shape: {self.data.shape}")
        self.logger.info('=' * 80)

    def save_results(self):
        """Save the output image and porosity information"""
        self._save_image()
        self._save_params()
