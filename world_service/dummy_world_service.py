"""
Unlike the real world service (which reads from world files), this one is
designed to test the block renderer in isolation.

Given a request for coordinate (x, y, z), it will return a single block with
block id x and metadata id y.
"""

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from lrpc.lrpc import get_lrpc
import asyncio
from voxelproto import voxelregion_pb2
from voxelproto import world_service_pb2
from voxelproto.voxelregion_pb2 import VoxelRegion
from voxelproto.world_service_pb2 import ListRequest, ListResponse
from voxelproto.world_service_pb2 import LoadRequest, LoadResponse
from voxelproto.world_service_pb2 import RegionRequest, RegionResponse
from voxelproto.world_service_pb2 import WorldServiceServicer

# Prefer to work with numpy arrays
import numpy as np
from util.monkeypatch_voxelregion import monkeypatch
monkeypatch()

PATHS = [
    "DUMMY"
]

lrpc = get_lrpc()

class DummyWorldServiceServicer(WorldServiceServicer):
    async def List(self, msg):
        print("Received message: List")
        resp = ListResponse()
        resp.paths.extend(PATHS)
        return resp

    async def Load(self, msg):
        print("Received message: Load")
        return LoadResponse()

    async def GetRegions(self, msg):
        print("Received message: Region")
        print("Sending region", msg.position)

        resp = RegionResponse()
        resp.regions.extend([VoxelRegion()])
        region = resp.regions[0]

        region.position.extend([0,0,0])
        region.dimensions.extend([1,1,1])

        x, y, z = msg.position

        data = np.zeros((1, 1, 1), dtype=np.uint32)
        data[0,0,0] = (y << 8) | x
        region.voxels_u32 = np.reshape(data, (-1,))
        return resp

lrpc.add_servicer(DummyWorldServiceServicer())

from util import ipyloop

lrpc.add_to_event_loop()
asyncio.get_event_loop().run_forever()
