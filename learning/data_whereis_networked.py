"""
This file can be used to train embeddings for scenes that come from Minecraft
save files. For our initial work, however, we do not use this file (and use
synthetic scenes only)

Note that unlike the rest of the files in this folder, this one requires LRPC
to be present on the current network (rather than reading data directly from
files on-disk.)
"""

# %cd ~/dev/mctest/learning
# import xdbg

#%%

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# %%

import tensorflow as tf
import numpy as np
import asyncio
import re
from data_whereis import DataGenerator

from lrpc.lrpc import get_lrpc
from voxelproto import world_service_pb2, snapshot_service_pb2
from util.monkeypatch_voxelregion import monkeypatch
monkeypatch()

# %%

class SampledWorldData(DataGenerator):
    def __init__(self, world_name='Varenburg', snapshots_paths=['/whereis3/'], capacity=100):
        super().__init__(capacity=capacity, unsupervised=True)
        self.world_name = world_name
        self.snapshots_paths = snapshots_paths

        self.class_counts = np.zeros(256, dtype=float)
        self.class_counts[0] = float('inf') # Never sample air!!!

        self.region_size = 25
        self.region_offset = self.region_size // 2
        self.nonborder_mask = np.zeros((self.region_size,)*3, dtype=bool)
        self.nonborder_mask[2:-2,2:-2,2:-2] |= True

    def prepare(self):
        self.lrpc = get_lrpc()
        self.snapshot_rpc = snapshot_service_pb2.SnapshotService(self.lrpc)
        self.world_rpc = world_service_pb2.WorldService(self.lrpc)
        self.lrpc.add_to_event_loop()

        self.snapshots = None
        self.snapshots_cache = None

    async def init_snapshots(self):
        # Load the correct world into the world server

        req = world_service_pb2.ListRequest()
        resp = await self.world_rpc.List(req)

        valid_paths = [path for path in resp.paths if re.search(self.world_name, path)]
        if not valid_paths:
            raise Exception("world_name regex matched no world paths")
        elif len(valid_paths) > 1:
            raise Exception("world_name regex matched more than one world path")

        req = world_service_pb2.LoadRequest()
        req.path = valid_paths[0]
        await self.world_rpc.Load(req)

        # Get a list of snapshots
        req = snapshot_service_pb2.FindSnapshots()
        req.paths.extend(self.snapshots_paths)

        resp = await self.snapshot_rpc.Find(req)
        self.snapshots = resp.snapshots

        self.snapshots_cache = [None] * len(self.snapshots)

    async def example_gen(self):
        if self.snapshots is None:
            await self.init_snapshots()

        candidates_loc = None
        while candidates_loc is None or not len(candidates_loc[0]) > 0:
            snapshot_num = np.random.choice(len(self.snapshots))
            if self.snapshots_cache[snapshot_num] is not None:
                voxels = self.snapshots_cache[snapshot_num]
            else:
                snapshot = self.snapshots[snapshot_num]

                req = world_service_pb2.RegionRequest()
                wx, wy, wz = [int(x) - self.region_offset for x in snapshot.position]
                req.position.extend([wx, wy, wz])
                req.dimensions.extend([self.region_size,]*3)

                resp = await self.world_rpc.GetRegions(req)

                voxels = resp.regions[0].voxels_u32.reshape((self.region_size,)*3)
                self.snapshots_cache[snapshot_num] = voxels

            interest_mask = self.get_interest_mask(voxels)
            candidates_loc = np.nonzero(interest_mask)

        candidates_p = -self.class_counts[voxels[candidates_loc] % 256]
        candidates_p = np.exp(candidates_p - np.max(candidates_p))
        candidates_p /= candidates_p.sum()

        voxel_idx = np.random.choice(len(candidates_p), p=candidates_p)
        voxel_loc = (xx, yy, zz) = np.array([candidates_loc[i][voxel_idx] for i in range(3)])

        voxels = voxels[xx-2:xx+3,yy-2:yy+3,zz-2:zz+3]
        example_id = "DUMMY2:9:9:9:0.75:-0.75:0"

        self.class_counts[voxels[2,2,2] % 256] += 1

        # print('gen one', len([x for x in self.snapshots_cache if x is not None]))
        return [{k: v for k,v in locals().items() if k in self.data_key_order}]

    def get_interest_mask(self, voxels):
        air_bool = (voxels == 0)
        air_adjacent = (
            air_bool[2:,1:-1,1:-1]
            | air_bool[:-2,1:-1,1:-1]
            | air_bool[1:-1,2:,1:-1]
            | air_bool[1:-1,:-2,1:-1]
            | air_bool[1:-1,1:-1,2:]
            | air_bool[1:-1,1:-1,:-2]
        )

        nonair_and_air_adjacent = np.zeros_like(voxels)
        nonair_and_air_adjacent[1:-1,1:-1,1:-1] = (voxels[1:-1,1:-1,1:-1] != 0) & air_adjacent

        return nonair_and_air_adjacent & self.nonborder_mask

def main():
    sess = tf.InteractiveSession()
    run_options = tf.RunOptions(timeout_in_ms=90000)

    dg = SampledWorldData()
    inputs = dg.get_inputs(batch_size=1)

    coord = tf.train.Coordinator()
    coord._dg_threads = dg.create_threads(sess, coord, start=True)
    coord._my_threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('requesting...')
    try:
        voxels_val, example_id_val = sess.run(inputs, options=run_options)
    except tf.errors.DeadlineExceededError:
        print('Deadline exceeded!')
        sys.exit(1)

    voxels_val = voxels_val[0]
    example_id_val = example_id_val[0]

    from task_tools import whereis_inspection
    task = whereis_inspection.get_task_from_values(
        voxels=voxels_val,
        words_str="Center voxel is {}:{}".format(
            voxels_val[2,2,2] & 0xff,
            voxels_val[2,2,2] >> 8).split(),
        candidates_mask=np.zeros_like(voxels_val),
        misty_location=[-1,-1,-1],
        example_id=example_id_val,
        )

    whereis_inspection.activate_task(task, from_values=True).result()

if __name__ == '__main__':
    main()
