import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

import lcmproto
import asyncio
from voxelproto.example_pb2 import ExampleMessage

def my_handler(channel, msg):
    print("Received message on channel \"%s\"" % channel)
    print("   timestamp   = %s" % str(msg.timestamp))
    print("   position    = %s" % str(msg.position))
    print("   orientation = %s" % str(msg.orientation))
    print("   ranges: %s" % str(msg.ranges))
    print("   name        = '%s'" % msg.name)
    print("   enabled     = %s" % str(msg.enabled))
    print("")

lc = lcmproto.LCMProto()
subscription = lc.subscribe("EXAMPLE", ExampleMessage, my_handler)

lc.add_to_event_loop()
asyncio.get_event_loop().run_forever()
