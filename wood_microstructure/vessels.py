"""Functions to generate/handle vessels in the wood microstructure."""
import numpy as np
import numpy.typing as npt

from .clocks import Clock

__all__ = [
    'generate_indexes',
    'filter_close',
    'filter_ray_close',
    'filter_edge',
    'extend',
]

@Clock('vessels')
def generate_indexes(vessel_count1:int, vessel_count2:int, lx: int, ly: int) -> npt.NDArray:
    """Generate vessel indexes.

    Args:
        vessel_count1 (int): Number of vessels in the first group
        vessel_count2 (int): Number of vessels in the second group
        num_x_nodes (int): Number of grid nodes in x direction
        num_y_nodes (int): Number of grid nodes in y direction

    Returns:
        npt.NDArray: Vessel x/y node_grid-indexes of shape (vessel_count, 2)
    """
    # Adjusted (-1) for 0-indexing
    x_rand_1 = np.round((np.random.rand(vessel_count1) * (lx - 16) + 8) / 2) * 2 - 2
    y_rand_1 = np.round((np.random.rand(vessel_count1) * (ly - 14) + 7) / 4) * 4 - 1
    y_rand_2 = np.round((np.random.rand(vessel_count2) * (ly - 14) + 7) / 2) * 2 - 1
    x_rand_2 = np.round((np.random.rand(vessel_count2) * (lx - 16) + 8) / 2) * 2 - 1

    x_rand_all = np.concatenate((x_rand_1, x_rand_2), axis=0)
    y_rand_all = np.concatenate((y_rand_1, y_rand_2), axis=0)
    vessel_all = np.column_stack((x_rand_all, y_rand_all))

    return vessel_all.astype(int)

@Clock('vessels')
def filter_close(vessel_all: npt.NDArray) -> npt.NDArray:
    """Filter the vessel that are too close to the other vessels

    Args:
        vessel_all (npt.NDArray): Vessel indexes to be filtered (shape: (vessel_count, 2))

    Returns:
        npt.NDArray: Filtered vessel indexes (shape: (new_vessel_count, 2))
    """
    all_idx = []
    done = set()
    for i, vessel in enumerate(vessel_all):
        if i in done:
            # This vessels was filtered because it was too close to another vessel
            continue
        dist = np.abs(vessel_all - vessel)
        mark0 = np.where((dist[:, 0] <= 6) & (dist[:, 1] <= 4))[0]
        all_idx.append(i)
        done.update(mark0)

    return vessel_all[all_idx]

@Clock('vessels')
def filter_ray_close(vessel_all: npt.NDArray, ray_cell_x_ind_all: npt.NDArray) -> npt.NDArray:
    """Filter the vessel that are too close to the ray cells

    Args:
        vessel_all (npt.NDArray): Vessel indexes to be filtered (shape: (vessel_count, 2))
        ray_cell_x_ind_all (npt.NDArray): Ray cell x indexes (shape: (ray_cell_count, 2))

    Returns:
        npt.NDArray: Filtered vessel indexes (shape: (new_vessel_count, 2))
    """
    all_idx = []
    for i, vessel in enumerate(vessel_all):
        diff = vessel[1] - ray_cell_x_ind_all
        if not np.any((diff >= -3) & (diff <= 4)):
            all_idx.append(i)
    return vessel_all[all_idx]

@Clock('vessels')
def filter_edge(vessel_all: npt.NDArray, lx: int, ly: int) -> npt.NDArray:
    """Filter the vessels that are too close to the edge of the grid.

    Args:
        vessel_all (npt.NDArray): Vessel indexes to be filtered (shape: (vessel_count, 2))
        lx (int): Number of grid nodes in x direction
        ly (int): Number of grid nodes in y direction

    Returns:
        npt.NDArray: Filtered vessel indexes (shape: (new_vessel_count, 2))
    """
    all_idx = []
    for i, vessel in enumerate(vessel_all):
        if vessel[0] < lx - 5 and vessel[1] < ly - 5:
            all_idx.append(i)
    return vessel_all[all_idx]

@Clock('vessels')
def extend(vessel_all: npt.NDArray, lx: int, ly: int) -> npt.NDArray:
    """The vessels are extended. Some vessels are extended to be double-vessel cluster, some are
    triple-vessel clusters.

    Args:
        vessel_all (npt.NDArray): Vessel indexes to be extended (shape: (vessel_count, 2))
        lx (int): Number of grid nodes in x direction
        ly (int): Number of grid nodes in y direction

    Returns:
        npt.NDArray: Extended vessel indexes (shape: (new_vessel_count, 2))
    """
    vessel_all_extend = np.empty((0, 2))
    for vessel in vessel_all:
        dist = vessel_all - vessel

        mark0 = np.where((dist[:, 0] <= 24) & (dist[:, 0] >= -8) & (np.abs(dist[:, 1]) <= 8))[0]
        mark1 = np.where((dist[:, 0] <= 12) & (dist[:, 0] >= -6) & (np.abs(dist[:, 1]) <= 6))[0]

        sign1 = np.random.choice([-1, 1])
        sign2 = np.random.choice([-1, 1])

        if len(mark0) > 1:
            vessel_all_extend = np.vstack((vessel_all_extend, vessel))
            possibility = np.random.rand(1)
            if len(mark1) <= 1:
                if possibility < 0.2:
                    temp = [vessel[0] + 6 + sign1, vessel[1] + sign2 * 2]
                    vessel_all_extend = np.vstack((vessel_all_extend, temp))
                elif possibility < 0.5:
                    temp = [vessel[0] + 6, vessel[1]]
                    vessel_all_extend = np.vstack((vessel_all_extend, temp))
        else:
            if vessel[0] + 12 < lx and vessel[1] + 10 < ly:
                temp0 = [vessel[0] + 5 + sign1, vessel[1]]
                possibility = np.random.rand(1)
                if possibility < 0.3:
                    temp = np.vstack((
                        temp0,
                        [temp0[0] + 5, temp0[1] + 2 * sign2]
                    ))
                else:
                    temp = np.vstack((
                        temp0,
                        [temp0[0] + 5 + sign2, temp0[1]]
                    ))
                vessel_all_extend = np.vstack((vessel_all_extend, vessel, temp))
            else:
                vessel_all_extend = np.vstack((vessel_all_extend, vessel))
    return vessel_all_extend

@Clock('vessels')
def get_grid_idx_in_vessel(vessel_all: npt.NDArray) -> npt.NDArray:
    """Get the the indexes of the grid nodes inside the vessels.

    Args:
        vessel_all (npt.NDArray): Vessel indexes (shape: (vessel_count, 2))

    Returns:
        npt.NDArray: Array of shape (vessel_count, 6, 2) with the indexes of the grid nodes inside the vessels.
                     Every vessel gives 6 x/y grid node indexes.
    """
    num_vess = vessel_all.shape[0]  # (num_vess, 2)

    indexes = np.empty((num_vess, 6, 2), dtype=int)
    indexes[:, :, :] = vessel_all[:, np.newaxis, :]
    indexes += [
        (-1, -2),
        (+1, -2),
        (-2, +0),
        (+2, +0),
        (-1, +2),
        (+1, +2)
    ]

    return indexes

@Clock('vessels')
def get_grid_idx_edges(vessel_all: npt.NDArray) -> npt.NDArray:
    """Get the indexes of the grid nodes at the edges of the vessels.

    Args:
        vessel_all (npt.NDArray): Vessel indexes (shape: (vessel_count, 2))

    Returns:
        npt.NDArray: Array of shape (vessel_count, 6, 2) with the indexes of the grid nodes at the edges of the vessels.
                     Every vessel gives 6 x/y grid node indexes.
    """
    num_vess = vessel_all.shape[0]  # (num_vess, 2)

    indexes = np.empty((num_vess, 6, 2), dtype=int)
    indexes[:, :, :] = vessel_all[:, np.newaxis, :]
    indexes += [
        (-3, -1),
        (-3, +1),
        (+0, -3),
        (+0, +3),
        (+3, -1),
        (+3, +1)
    ]

    return indexes
