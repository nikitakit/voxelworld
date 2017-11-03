
# %cd ~/dev/mctest/task_tools

# import xdbg

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# %%

from util.monkeypatch_voxelregion import monkeypatch
monkeypatch()
import util.struct_tools
from voxelproto.voxelregion_pb2 import VoxelRegion
import math

import numpy as np
from util import transformations

from voxelproto import voxelregion_pb2
from voxelproto import world_service_pb2
from voxelproto.crowd_task_pb2 import CrowdTask
from voxelproto.task_service_pb2 import TaskServiceRequest, NamedTask
from voxelproto.world_service_pb2 import RegionRequest
from voxelproto.world_snapshot_pb2 import WorldSnapshot

# %%

# with open("tasks/internal2/guess1.pb", 'rb') as f:
#     task = NamedTask.FromString(f.read())
#
# snapshot = task.processed_task.data_static_world.snapshot
#
# %%
# region = snapshot.regions[0]
# position = region.position
# voxels = np.reshape(region.voxels_u32, region.dimensions)
#
# snapshot.position

# snapshot.rotation

def get_camera_vector(rotation):
    rotation_mtx = (
        transformations.rotation_matrix(rotation[0], [1,0,0]) @
        transformations.rotation_matrix(rotation[1], [0,1,0]) @
        transformations.rotation_matrix(rotation[2], [0,0,1])
    )

    return -np.linalg.inv(rotation_mtx[:3, :3]) @ np.array([0, 0, 1.])


def get_viewport_mask_relative(voxels, camera_position, rotation):
    """
    Return a mask indicating which voxels are visible from a given vantage
    point.

    Uses an approximation of the camera viewport, because this actually changes
    depending on browser window size.
    """

    HALF_ANGLE_HORIZ = 0.4
    HALF_ANGLE_VERT = 0.24

    camera_position = np.asarray(camera_position)
    rotation_mtx = (
        transformations.rotation_matrix(rotation[0], [1,0,0]) @
        transformations.rotation_matrix(rotation[1], [0,1,0]) @
        transformations.rotation_matrix(rotation[2], [0,0,1])
    )

    mask = np.zeros_like(voxels, dtype=bool)

    # Note that the voxel the camera is in will be treated as invisible under
    # this algorithm.
    for x in range(voxels.shape[0]):
        for y in range(voxels.shape[1]):
            for z in range(voxels.shape[2]):
                ray = np.array([x,y,z]) + 0.5 - camera_position
                # Negation is due to camera coordinate space not being the same
                # as voxel coordinate space
                # TODO(nikita): consider figuring out how to transform the
                # snapshot schema such that this hack is not necessary.
                ray = -ray

                ray = rotation_mtx[:3,:3] @ ray

                angle_xz = math.atan2(ray[0], ray[2])
                if abs(angle_xz) > HALF_ANGLE_HORIZ:
                    continue

                angle_yz = math.atan2(ray[1], ray[2])
                if abs(angle_yz) > HALF_ANGLE_VERT:
                    continue

                mask[x,y,z] = True

    return mask


def raycast_relative(voxels, u, v):
    """
    Returns the coordinates of the first non-empty voxel hit when raycasting
    from position u in the direction of vector v

    See http://www.cse.chalmers.se/edu/year/2010/course/TDA361/grid.pdf

    PERF(nikita): this is a primary candidate for optimization, e.g. by
    switching to cython or numba. The paper describes the algorithm in terms of
    individual variables and not length-3 arrays, which should be faster because
    it uses less levels of indirection.
    """
    # Consider a raycast vector parametrized by u + t*v
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Our position in integer voxel-space
    pos = np.asarray(u, dtype=int)

    # Determines which direction our ray is marching in
    step = np.array(np.sign(v), dtype=int)

    # Change in t required to traverse one voxel in each direction
    t_delta = np.abs(1. / v)

    # Location of nearest voxel boundary, in units of t
    t_max = np.where(t_delta < np.inf,
                     t_delta * np.where(step > 0, pos + 1 - u, u - pos),
                     np.inf)

    while True:
        if voxels[pos[0], pos[1], pos[2]]:
            return pos
        min_axis = np.argmin(t_max)
        pos[min_axis] += step[min_axis]
        if pos[min_axis] < 0 or pos[min_axis] >= voxels.shape[min_axis]:
            break
        t_max[min_axis] += t_delta[min_axis]

    return None

#%%

def visibility_mask_relative(voxels, position, region_of_interest=None):
    """
    Get a mask of voxels that are visible from a given position.
    To only test a subset of the full voxel grid, specify a region_of_interest

    This does not consider the camera viewport, i.e. it does a raycast in all
    directions from the camera position.

    Args:
    - voxels: 3D numpy array (use 0 or False to indicate empty)
    - position: camera location (length-3 float iterable)
    - region_of_interest: (optional) bool numpy array of same shape as voxels

    Returns:
    - bool numpy array of same shape as voxels
    """

    position = np.asarray(position)
    voxels_bool = (voxels != 0)
    mask = np.zeros_like(voxels, dtype=bool)

    corner_offsets = list(np.array([
        [0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1],
        [0.1, 0.9, 0.1],
        [0.1, 0.1, 0.9],
        [0.9, 0.9, 0.1],
        [0.9, 0.1, 0.9],
        [0.1, 0.9, 0.9],
        [0.9, 0.9, 0.9],
    ]))

    for x in range(voxels.shape[0]):
        for y in range(voxels.shape[1]):
            for z in range(voxels.shape[2]):
                # Test visibility of block (x,y,z):
                # first set the voxel to not-transparent, so it absorbs ray hits
                if region_of_interest is not None and not region_of_interest[x,y,z]:
                    continue

                old_voxel = voxels_bool[x,y,z]
                voxels_bool[x,y,z] = True

                for offset in corner_offsets:
                    hit = raycast_relative(voxels_bool, position, np.array([x,y,z]) + offset - position)
                    assert hit is not None
                    if np.all(hit == np.array([x,y,z], dtype=int)):
                        mask[x,y,z] = True
                        break

                voxels_bool[x,y,z] = old_voxel
    return mask

# %%

def misty_locations_relative(voxels, position, rotation):
    """
    Returns possible misty locations.

    Args:
    - voxels: 3D numpy array
    - position: camera position relative to origin of this voxel region
    - rotation: camera rotation (in the wierd snapshot coordinate space)
    """
    voxels_bool = (voxels != 0)
    # Only consider air blocks that are adjacent to a non-air block
    has_adjacent = (
        voxels_bool[2:,1:-1,1:-1]
        | voxels_bool[:-2,1:-1,1:-1]
        | voxels_bool[1:-1,2:,1:-1]
        | voxels_bool[1:-1,:-2,1:-1]
        | voxels_bool[1:-1,1:-1,2:]
        | voxels_bool[1:-1,1:-1,:-2]
    )

    air_and_has_adjacent = np.zeros_like(voxels_bool)
    air_and_has_adjacent[1:-1,1:-1,1:-1] = (voxels_bool[1:-1,1:-1,1:-1] != True) & has_adjacent

    # Further restrict to only blocks within the camera viewport
    voxels_roi = get_viewport_mask_relative(voxels_bool, position, rotation)
    voxels_roi &= air_and_has_adjacent

    mask = visibility_mask_relative(voxels_bool, position, voxels_roi)
    return mask

def misty_locations_from_snapshot(snapshot):
    misty_locations = []
    print("Started searching for possible misty locations")
    for region in snapshot.regions:
        voxels = np.reshape(region.voxels_u32, region.dimensions)
        mask = misty_locations_relative(
            voxels,
            np.asarray(snapshot.position) - np.asarray(region.position),
            snapshot.rotation
            )
        for x, y, z in list(zip(*np.nonzero(mask))):
            misty_locations.extend([
                int(x + region.position[0]),
                int(y + region.position[1]),
                int(z + region.position[2]),
                ])
    assert len(misty_locations) % 3 == 0
    print("Locations found:", len(misty_locations) // 3)
    return misty_locations
