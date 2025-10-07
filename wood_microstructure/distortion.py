import numpy as np
import numpy.typing as npt


def get_distortion_grid(
        x_center: int, y_center: int, x_size: int, y_size: int, cutoff: int
    ) -> tuple[npt.NDArray, npt.NDArray]:
    """Get the X,Y grids centered on (x_center, y_center) with cutoff of `local_distortion_cutoff`"""
    x_grid, y_grid = np.mgrid[
        max(0, x_center - cutoff):min(x_size, x_center + cutoff),
        max(0, y_center - cutoff):min(y_size, y_center + cutoff)
    ].astype(int)

    return x_grid, y_grid

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
