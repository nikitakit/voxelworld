from concurrent import futures
import time

import grpc

import lcmbridge_pb2
import lcm
import queue

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

lc = lcm.LCM("udpm://228.6.7.8:7667")
msg_queue = queue.Queue()
send_channels = set()

def handle_msg(channel, msg):
    if channel in send_channels:
        return
    msg_queue.put((channel, msg))

lc.subscribe('.*', handle_msg)

class LCMBridge(lcmbridge_pb2.LCMBridgeServicer):
    def Send(self, request_iterator, context):
        for data in request_iterator:
            print('Client->Server on channel', data.channel)
            send_channels.add(data.channel)
            lc.publish(data.channel, data.msg)

        return lcmbridge_pb2.Empty()

    def Receive(self, request, context):
        # Override message queue so all new messages are re-routed
        global msg_queue
        msg_queue = queue.Queue()
        while True:
            channel, msg = msg_queue.get()
            print('Server->Client on channel', channel)
            yield lcmbridge_pb2.LCMMessage(channel=channel, msg=msg)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    lcmbridge_pb2.add_LCMBridgeServicer_to_server(LCMBridge(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("LCMBridge server started")
    try:
        while True:
            lc.handle()
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
