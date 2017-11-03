"""
LCMProto WebSocket Bridge: server application

Usage:
  python lcm_ws_bridge.py --descriptor=path/to/descriptor.desc [--port=8000]

This code is based on the original LCM Websocket bridge, which is part of
https://github.com/pioneers/forseti2
"""
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.httpserver
import json
import os
import sys
import watchdog.observers
import watchdog.events

from __init__ import LCMProto
import copy
import google.protobuf.descriptor_pb2
import google.protobuf.json_format
from google.protobuf.message_factory import MessageFactory
from google.protobuf.symbol_database import Default as DefaultSymbolDatabase
from google.protobuf.descriptor_pb2 import FileDescriptorProto

from tornado.options import define, options

define('port', default=8000, help='run on the given port', type=int)
define('descriptor', default="",
       help='descriptor file that specifies protobuf definitions', type=str)

tornado.options.parse_command_line()
if not options.descriptor:
    print("ERROR: need descriptor")
    sys.exit(1)
descriptor_path = os.path.expanduser(options.descriptor)

# Importing protobuf well-known types is required to make them available later
import google.protobuf.struct_pb2
import google.protobuf.timestamp_pb2

# Save the list of file descriptors now, so it doesn't change by accident
file_descriptors = copy.copy(DefaultSymbolDatabase().pool._file_descriptors)

message_classes = {}
method_info = {}

def update_message_classes():
    global message_classes, descriptor_path, method_info
    factory = MessageFactory()
    # Add well-known types first
    for file_descriptor in file_descriptors.values():
        file_proto = FileDescriptorProto()
        file_proto.ParseFromString(file_descriptor.serialized_pb)
        factory.pool.Add(file_proto)
    # Then add our types
    with open(descriptor_path, 'rb') as f:
        fileset = google.protobuf.descriptor_pb2.FileDescriptorSet.FromString(f.read())
    for file_proto in fileset.file:
        factory.pool.Add(file_proto)
    message_classes = factory.GetMessages([file_proto.name for file_proto in fileset.file])

    # HACK to add nested types. Is there an API for this?
    for desc in factory.pool._descriptors.values():
        if desc.full_name not in message_classes:
            message_classes[desc.full_name] = factory.GetPrototype(desc)

    method_info = {}

    for file_proto in fileset.file:
        for service in file_proto.service:
            for method in service.method:
                k = "{}.{}".format(service.name, method.name)
                input_type = method.input_type
                output_type = method.output_type
                if input_type.startswith('.'):
                    input_type = input_type[1:]
                if output_type.startswith('.'):
                    output_type = output_type[1:]
                if input_type not in message_classes or output_type not in message_classes:
                    print("WARNING: types for method {} not found".format(k))
                input_type = message_classes[input_type]
                output_type = message_classes[output_type]

                method_info[k] = (method, input_type, output_type)


update_message_classes()

lc = LCMProto()

class WSHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        """
        Called when a client opens the websocket
        """
        print("open called")
        self.lc = lc
        self.subscriptions = {}
        self.active_calls = {}
        self.defined_methods = {}

    def on_close(self):
        """
        Called when the websocket closes
        """
        print("closed called")
        for subscription_id in list(self.subscriptions.keys()):
            self.remove_subscription(subscription_id)
        for uid, val in self.active_calls.items():
            self.lc.unsubscribe(val)
        for method_name, val in self.defined_methods.items():
            self.lc.unsubscribe(val)

    ### Websocket-related

    def on_message(self, message):
        """
        Called when a message is received over the websocket
        """

        obj = json.loads(message)
        msg_type = obj["type"]
        data = obj["data"]

        if msg_type == "subscribe":
            self.add_subscription(data["channel"],
                                  data["msg_type"],
                                  data["subscription_id"])
        elif msg_type == "unsubscribe":
            self.remove_subscription(data["subscription_id"])
        elif msg_type == "publish":
            self.publish(data["channel"], data["msg_type"], data["data"])
        elif msg_type == "call": # for LRPC
            self.call(data["method"],
                      data["uid"],
                      data["data"])
        elif msg_type == "server_define":
            self.server_define(data["method"])
        elif msg_type == "server_response":
            self.server_response(data["method"], data["uid"], data["data"])
        else:
            raise Exception("Invalid websocket message type: " + msg_type)

    def ws_send(self, type, data):
        """
        Convenience method for sending data over the websocket
        """
        self.write_message(json.dumps({"type": type, "data": data}))

    ### LCM-related

    def add_subscription(self, channel, msg_type, subscription_id):
        """
        Creates an LCM subscription (based on data from a websocket request)
        Forwards any LCM messages received to javascript via websockets
        """
        global message_classes
        if msg_type not in message_classes:
            print("WARNING: subscription with invalid message type was dropped")
            return

        msg_type = message_classes[msg_type]

        def handle(channel, msg):
            self.ws_send("packet", {"subscription_id": subscription_id,
                                    "msg": json.loads(google.protobuf.json_format.MessageToJson(
                                        msg,
                                        including_default_value_fields=True))})
        self.subscriptions[subscription_id] = self.lc.subscribe(channel, msg_type, handle)

    def remove_subscription(self, subscription_id):
        if subscription_id not in self.subscriptions:
            return
        print("UNSUBSCRIBING")
        self.lc.unsubscribe(self.subscriptions[subscription_id])
        del self.subscriptions[subscription_id]

    def publish(self, channel, msg_type, data):
        global message_classes
        if msg_type not in message_classes:
            print("WARNING: publish with invalid message type was dropped")
            return

        msg = message_classes[msg_type]()
        google.protobuf.json_format.Parse(json.dumps(data), msg)
        self.lc.publish(channel, msg)

    def call(self, method, uid, data):
        global method_info
        if method not in method_info:
            print("WARNING: call to an unknown method was dropped")
            return

        method_data, input_type, output_type = method_info[method]

        def handle(channel, msg):
            if "/A/" in channel:
                self.ws_send("call_response", { "uid": uid, "msg": None})
                del self.active_calls[uid]
                self.lc.unsubscribe(subscription)
                return

            msg = output_type.FromString(msg)
            msg = google.protobuf.json_format.MessageToJson(
                msg, including_default_value_fields=True)
            msg = json.loads(msg)

            self.ws_send("call_response", { "uid": uid, "msg": msg})

            if not method_data.server_streaming:
                del self.active_calls[uid]
                self.lc.unsubscribe(subscription)


        subscription = super(LCMProto, self.lc).subscribe("{}/[aA]/{}".format(method, uid), handle)

        msg = input_type()
        google.protobuf.json_format.Parse(json.dumps(data), msg)
        self.lc.publish("{}/q/{}".format(method, uid), msg)
        self.active_calls[uid] = subscription

    def _handle_request(self, channel, msg):
        method, code, uid = channel.split('/')
        if method not in self.defined_methods:
            return

        if method not in method_info:
            print("WARNING: method info no longer available for existing definition")
            return

        method_data, input_type, output_type = method_info[method]

        msg = input_type.FromString(msg)
        msg = google.protobuf.json_format.MessageToJson(
            msg, including_default_value_fields=True)
        msg = json.loads(msg)

        self.ws_send("server_request", { "method": method, "uid": uid, "msg": msg})

    def server_define(self, method):
        global method_info
        if method not in method_info:
            print("WARNING: attempt to define unknown method was dropped")
            return

        subscription = super(LCMProto, self.lc).subscribe("{}/[qQ]/[A-Za-z0-9]+".format(method), self._handle_request)
        self.defined_methods[method] = subscription

    def server_response(self, method, uid, data):
        global method_info
        if method not in method_info:
            print("WARNING: call to an unknown method was dropped")
            return

        method_data, input_type, output_type = method_info[method]

        if method_data.server_streaming and data is None:
            self.lc.publish("{}/A/{}".format(method, uid), b'closed')
            return

        msg = output_type()
        google.protobuf.json_format.Parse(json.dumps(data), msg)
        self.lc.publish("{}/a/{}".format(method, uid), msg)


application = tornado.web.Application([
    (r'/', WSHandler)
])

def handler(*args):
    lc.handle()

class DescriptorModifiedHandler(watchdog.events.PatternMatchingEventHandler):
    def on_modified(self, info):
        print('Protobuf definitions updated')
        update_message_classes()

observer = watchdog.observers.Observer()
observer.schedule(DescriptorModifiedHandler([descriptor_path]), os.path.dirname(descriptor_path))

if __name__ == '__main__':
    observer.start()
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)
    loop = tornado.ioloop.IOLoop.instance()
    loop.add_handler(lc.fileno(), handler, tornado.ioloop.IOLoop.READ)
    print('Running LCMProto Websocket Bridge on port {}'.format(options.port))
    tornado.ioloop.IOLoop.instance().start()
