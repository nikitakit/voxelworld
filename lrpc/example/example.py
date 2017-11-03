import sys
sys.path.append('/Users/kitaev/dev/mctest/util/')
sys.path.append('/Users/kitaev/dev/lrpc/')
import ipyloop
from lrpc import get_lrpc
import asyncio

# %%

sys.path.append('/Users/kitaev/dev/lrpc/example/')
from example_service_pb2 import *

# %%

l = get_lrpc()
l.add_to_event_loop()

# %%

class MyServicer(ExampleServiceServicer):
    async def ExampleMethod(self, req):
        print("GOT request")
        print(req)
        self._last_req = req

        resp = ExampleReply()
        resp.enabled = req.enabled
        return resp

    async def ExampleStreamingOut(self, req, out):
        print("GOT request for streaming")
        print(req)
        self._last_req = req

        resp = ExampleReply()
        resp.enabled = req.enabled
        out.send(resp)
        resp.enabled = False
        out.send(resp)

# %%

l.add_servicer(MyServicer())

# %%

async def main():
    svc = ExampleService(l)

    req = ExampleRequest()
    req.enabled = True

    fut = svc.ExampleMethod(req)
    print('got result', await fut)

    resp = svc.ExampleStreamingOut(req)
    while True:
        val = await resp.recv()
        if val is None:
            break
        print('got val')
        print(val)
    print('done getting vals')

# %%

asyncio.get_event_loop().run_until_complete(main())
