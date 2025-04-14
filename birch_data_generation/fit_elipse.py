import numpy as np
import numpy.typing as npt


def fit_elipse(point_coord: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Fit an ellipse to a set of points.

    Parameters
    ----------
    point_coord : npt.NDArray
        An array of shape (..., 4, 2) containing the x/y coordinates of the points to fit as columns.
        The ... assumes any number of leading dimensions to allow for broadcasting.
        See https://numpy.org/doc/2.2/reference/generated/numpy.linalg.solve.html

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
        4 NDarrays of shape (...), representing:
        - r1: semi-major axis length
        - r2: semi-minor axis length
        - h: x-coordinate of the center
        - k: y-coordinate of the center
    """
    A = np.concatenate((point_coord**2, point_coord), axis=-1)
    b = np.ones(A.shape[-1])

    x = np.linalg.solve(A, b)
    x *= np.sign(x[..., 0])[..., np.newaxis]

    h = -x[..., 2] / x[..., 0] / 2
    k = -x[..., 3] / x[..., 1] / 2

    diff = np.abs(x[..., 0] - x[..., 1])

    r1 = np.sqrt(2 * (x[...,0] * h**2 + x[...,1] * k**2 - 1) / (x[...,0] + x[...,1] + diff))
    r2 = np.sqrt(2 * (x[...,0] * h**2 + x[...,1] * k**2 - 1) / (x[...,0] + x[...,1] - diff))

    return r1, r2, h, k

def fit_ellipse_6pt(point_coord: npt.NDArray) -> tuple[float, float, float, float]:
    """Fit 6 points to an ellipse."""
    assert len(point_coord.shape) == 2
    assert point_coord.shape[1] == 2
    assert point_coord.shape[0] == 6

    point_coord = np.concatenate((point_coord**2, point_coord), axis=-1)
    b = np.ones(6)

    x, _, _, _ = np.linalg.lstsq(point_coord, b, rcond=None)

    x *= np.sign(x[0])

    h = -x[2] / x[0] / 2
    k = -x[3] / x[1] / 2

    diff = np.abs(x[0] - x[1])

    r1 = np.sqrt(2 * (x[0] * h**2 + x[1] * k**2 - 1) / (x[0] + x[1] + diff))
    r2 = np.sqrt(2 * (x[0] * h**2 + x[1] * k**2 - 1) / (x[0] + x[1] - diff))

    return r1, r2, h, k
