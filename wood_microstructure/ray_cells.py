"""Functions to generate/handle ray cells in the wood microstructure."""
import numpy as np
import numpy.typing as npt


def get_x_indexes(ly: int, trim: int, ray_space: float, random_width: float = 10) -> npt.NDArray:
    """Get the x indexes of the ray cells.

    Args:
        ly (int): Number of grid nodes in y direction
        trim (int): Trim `trim` number of grid nodes from the start and end
        ray_space (float): Distance between ray cells
        space (float): Space between ray cells
        random_width (float): Randomiazion of the ray cell width (+/- random_width / 2)

    Returns:
        npt.NDArray: X indexes of the ray cells
    """
    rwh = random_width / 2
    ray_cell_linspace = np.arange(trim - 1, ly - trim, ray_space)
    indexes = ray_cell_linspace + np.random.rand(len(ray_cell_linspace)) * random_width - rwh
    indexes = np.floor(indexes / 2) * 2

    return indexes.astype(int)

def distribute(
        sie_z: int, indexes: npt.NDArray, cell_num: float, cell_num_std: float, height: float,
        height_mod: int
    ) -> tuple[npt.NDArray, list[npt.NDArray], npt.NDArray]:
    """Distribute the ray cells across the volume

    Args:
        sie_z (int): Size of the volume in z direction
        indexes (npt.NDArray): Ray cell indices
        cell_num (float): number of cells
        cell_num_std (float): standard deviation of the number of cells
        height (float): height of the ray cells
        height_mod (int): height modifier

    Returns:
        tuple[npt.NDArray, list[npt.NDArray]]:
        - Ray cell indices (num_ray_cells, 2): indices array A where A[j][1] = A[j][0] + 1
        - Ray cell widths (num_ray_cells, non_uniform): length of elements depends on the randomly generated group
    """
    x_ind = []
    width = []

    m = int(np.ceil(sie_z / cell_num / height + 6))
    for idx in indexes:
        app = [0]
        ray_cell_space = np.round(16 * np.random.rand(m)) + height_mod
        rnd = np.round(-30 * np.random.rand())
        # ray_idx = [idx, idx + 1]
        for rs in ray_cell_space:
            group = np.random.randn() * cell_num_std + cell_num
            group = np.clip(group, 5, 25)
            app = app[-1] + (np.arange(group + 1) + rs + rnd) * height
            rnd = 0

            if app[0] > sie_z - 150:
                break

            if app[-1] >= 150:  # TODO should 150 be a parameter indicating the start of the ray cell?
                # x_ind.append(ray_idx)
                x_ind.append(idx)
                width.append(np.round(app).astype(int))

    return np.array(x_ind, dtype=int), width
