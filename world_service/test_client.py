import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from lrpc.lrpc import get_lrpc
import asyncio
from voxelproto import voxelregion_pb2
from voxelproto import world_service_pb2
from voxelproto.world_service_pb2 import WorldService, ListRequest, LoadRequest, RegionRequest

async def main():
    print("[main] started")
    req = ListRequest()
    reply = await world_service.List(req)
    print('[main] got path list', reply.paths)

    req = LoadRequest()
    req.path = reply.paths[0]
    reply = await world_service.Load(req)
    print('[main] loaded world')

    req = RegionRequest()
    req.dimensions.extend((2,2,2))
    req.position.extend((-351,10,372))
    reply = await world_service.GetRegions(req)
    print('[main] got region', reply.regions[0].voxels_u32.voxels_u32[0])


lrpc = get_lrpc()
world_service = WorldService(lrpc)

lrpc.add_to_event_loop()
# from util import ipyloop
asyncio.get_event_loop().run_until_complete(main())
