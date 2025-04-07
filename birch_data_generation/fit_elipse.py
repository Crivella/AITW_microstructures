# import time

import numpy as np


def fit_elipse(point_coord):
    """
    Fit an ellipse to a set of points.

    Parameters
    ----------
    point_coord : np.ndarray
        A 2D array of shape (n_points, 2) containing the coordinates of the points.

    Returns
    -------
    np.ndarray
        A 1D array containing the parameters of the fitted ellipse.
    """
    # start = time.time()
    A = np.column_stack((point_coord**2, point_coord))
    b = np.ones(A.shape[0])

    # print(point_coord)
    # print('A:', A.shape, A)

    if A.shape[0] == 4:
        x = np.linalg.solve(A, b)
    else:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)[0]

    # print('fit ellipse parameters:', x)
    # print(A @ x, b)
    x *= np.sign(x[0])


    h = -x[2] / x[0] / 2
    k = -x[3] / x[1] / 2

    # print('h, k:', h, k)

    # print(x[0] + x[1])
    # print(abs(x[0] - x[1]))

    # r1 = sqrt(2*(x(1)*h^2+x(2)*k^2-1)/(x(1)+x(2)+abs(x(1)-x(2))));
    # r2 = sqrt(2*(x(1)*h^2+x(2)*k^2-1)/(x(1)+x(2)-abs(x(1)-x(2))));

    r1 = np.sqrt(2 * (x[0] * h**2 + x[1] * k**2 - 1) / (x[0] + x[1] + abs(x[0] - x[1])))
    r2 = np.sqrt(2 * (x[0] * h**2 + x[1] * k**2 - 1) / (x[0] + x[1] - abs(x[0] - x[1])))
    # print('r1, r2:', r1, r2)

    # print('fit ellipse time:', time.time() - start)

    return r1, r2, h, k
