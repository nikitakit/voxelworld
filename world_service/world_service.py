import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from lrpc.lrpc import get_lrpc
import asyncio
import concurrent.futures
from voxelproto import voxelregion_pb2
from voxelproto import world_service_pb2
from voxelproto.voxelregion_pb2 import VoxelRegion
from voxelproto.world_service_pb2 import ListRequest, ListResponse
from voxelproto.world_service_pb2 import LoadRequest, LoadResponse
from voxelproto.world_service_pb2 import RegionRequest, RegionResponse
from voxelproto.world_service_pb2 import WorldServiceServicer

from mceditlib.worldeditor import WorldEditor

# Prefer to work with numpy arrays
import numpy as np
from util.monkeypatch_voxelregion import monkeypatch
monkeypatch()

PATHS = [
    "/Users/kitaev/Downloads/Stranded",
    "/Users/kitaev/Documents/MinecraftMaps/Varenburg 1.1.1/Varenburg 1.1.1"
]

editor = None
dimension = None

class WorldServiceServicerImpl(WorldServiceServicer):
    async def List(self, msg):
        global PATHS
        print("Received message: List")
        resp = ListResponse()
        resp.paths.extend(PATHS)
        return resp

    async def Load(self, msg):
        print("Received message: Load")
        global PATHS, editor, dimension

        if msg.path not in PATHS:
            return

        print("Loading world:", msg.path)
        if editor is not None:
            dimension = None
            editor.close()

        editor = WorldEditor(msg.path, readonly=True)
        dimension = editor.getDimension()

        return LoadResponse()

    async def GetRegions(self, msg):
        print("Received message: Region", msg.position)
        resp = await asyncio.get_event_loop().run_in_executor(executor, self.get_regions, msg)
        print("Responding with region", msg.position)
        return resp

    def get_regions(self, msg):
        global dimension
        block_id = dimension.getBlock(*msg.position).ID

        resp = RegionResponse()
        resp.regions.extend([VoxelRegion()])
        region = resp.regions[0]

        region.position.extend(msg.position)
        region.dimensions.extend(msg.dimensions)

        x, y, z = msg.position
        length, height, depth = msg.dimensions

        data = np.zeros((length, height, depth), dtype=np.uint32)
        # Fill in data with naive iteration. TODO: consider speed-ups
        for i in range(x, x+length):
            for j in range(y, y+height):
                for k in range(z, z+depth):
                    data[i-x,j-y,k-z] = (dimension.getBlockData(i,j,k) << 8) | dimension.getBlockID(i,j,k)

        region.voxels_u32 = np.reshape(data, (-1,))
        return resp


lrpc = get_lrpc()
lrpc.add_servicer(WorldServiceServicerImpl())
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

from util import ipyloop

lrpc.add_to_event_loop()
asyncio.get_event_loop().run_forever()
