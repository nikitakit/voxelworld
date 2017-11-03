import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

import lcmproto
import time

from voxelproto.example_pb2 import ExampleMessage

lc = lcmproto.LCMProto()
channel = lc.channel("EXAMPLE", ExampleMessage)

msg = ExampleMessage()
msg.timestamp = int(time.time() * 1000000)
del msg.position[:]
msg.position.extend((1, 2, 3))
del msg.orientation[:]
msg.orientation.extend((1, 0, 0, 0))
del msg.ranges[:]
msg.ranges.extend(range(15))
msg.name = "example string"
msg.enabled = True

channel.publish(msg)
