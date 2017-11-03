import grpc
import time
import threading
import lcm

import lcmbridge_pb2

lc = lcm.LCM("udpm://228.6.7.8:7667")
msg_queue = []
recv_channels = set()

def handle_msg(channel, msg):
    if channel in recv_channels:
        return
    msg_queue.append((channel, msg))

lc.subscribe('.*', handle_msg)

def sender_iter():
    while True:
        lc.handle()
        for (channel, msg) in msg_queue:
            print('Client->Server on channel', channel)
            yield lcmbridge_pb2.LCMMessage(channel=channel, msg=msg)
        msg_queue.clear()

def recv(stub):
    for data in stub.Receive(lcmbridge_pb2.Empty()):
        print('Server->Client on channel', data.channel)
        recv_channels.add(data.channel)
        lc.publish(data.channel, data.msg)

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = lcmbridge_pb2.LCMBridgeStub(channel)

    recv_thread = threading.Thread(target=recv, args=(stub,), daemon=True)
    recv_thread.start()

    print("LCMBridge client started")
    stub.Send(sender_iter())

if __name__ == '__main__':
    run()
