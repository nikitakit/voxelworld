"""
LRPC - python
"""

import asyncio
import lcm
import functools
import uuid
import base64

# %%

class SingleSender:
    def __init__(self, lc, name, uid, msg_type):
        self.lc = lc
        self.channel = "{}/a/{}".format(name, uid)
        self.msg_type = msg_type
        self.closed = False

    def send(self, msg):
        if self.closed:
            raise Exception("Attempt to send on closed stream")

        if not isinstance(msg, self.msg_type):
            raise ValueError("Invalid message type: {}", type(msg))

        self.lc.publish(self.channel, msg.SerializeToString())
        self.closed = True

    async def _call(self, func, msg):
        result = await func(msg)
        if result is not None:
            if self.closed:
                raise Exception("Attempt to send on closed stream")
            self.send(result)
        elif not self.closed:
            raise Exception("Closing a stream without sending a response")

class StreamSender:
    def __init__(self, lc, name, uid, msg_type):
        self.lc = lc
        self.name = name
        self.uid = uid
        self.msg_type = msg_type
        self.closed = False

    def send(self, *msgs):
        if self.closed:
            raise Exception("Attempt to send on closed stream")

        for msg in msgs:
            if not isinstance(msg, self.msg_type):
                raise ValueError("Invalid message type: {}", type(msg))

            self.lc.publish(self.name + "/a/" + self.uid, msg.SerializeToString())

    def close(self):
        if not self.closed:
            self.lc.publish(self.name + "/A/" + self.uid, b'closed')
            self.closed = True

    async def _call(self, func, msg):
        return_val = await func(msg, self)
        if return_val is not None:
            print("WARNING: ignored return value from call to", func)
        self.close()

class ActiveCall:
    def __init__(self):
        self.fut_list = []
        self.queue = []

    def recv(self):
        fut = asyncio.Future()

        if self.queue is None:
            fut.set_exception(Exception("Attempt to read from closed stream"))
        elif self.queue:
            val = self.queue.pop(0)
            if val is None:
                self.queue = None
            fut.set_result(val)
        else:
            self.fut_list.append(fut)

        return fut

class LRPC:
    instance = None

    def __init__(self, lc, loop):
        self.lc = lc
        self.loop = loop
        self._added_to_event_loop = False
        self._channel_to_futures = {}
        self._channel_to_queue = {}
        self._name_to_type = {}
        self._idx = 0 # XXX: TODO(nikita): use real uids

    def run_forever(self):
        if not self._added_to_event_loop:
            self.add_to_event_loop()

        self.loop.run_forever()

    def add_servicer(self, servicer):
        for (channel_prefix,
             bound_method,
             input_type, output_type,
             client_streaming, server_streaming) in servicer._get_manifest():
            if not client_streaming and not server_streaming:
                def handler(channel_prefix, bound_method, input_type, output_type, channel, data):
                    msg = input_type.FromString(data)

                    code, uid = self._unpack_channel(channel)
                    out = SingleSender(self.lc, channel_prefix, uid, output_type)
                    asyncio.ensure_future(out._call(bound_method, msg), loop=self.loop)
                self._subscribe(functools.partial(handler, channel_prefix, bound_method, input_type, output_type),
                    channel_prefix, "q")
            elif not client_streaming and server_streaming:
                def handler(channel_prefix, bound_method, input_type, output_type, channel, data):
                    msg = input_type.FromString(data)

                    code, uid = self._unpack_channel(channel)
                    out = StreamSender(self.lc, channel_prefix, uid, output_type)
                    asyncio.ensure_future(out._call(bound_method, msg), loop=self.loop)
                self._subscribe(functools.partial(handler, channel_prefix, bound_method, input_type, output_type),
                    channel_prefix, "q")
            else:
                raise NotImplementedError("Streaming input is not yet implemented")

    def add_stub(self, stub, manifest):
        for (channel_prefix,
             method_name,
             input_type, output_type,
             client_streaming, server_streaming) in manifest:
            self._name_to_type[channel_prefix] = output_type

            if not client_streaming and not server_streaming:
                # TODO(nikita): set the name correctly
                def method(channel_prefix, input_type, req):
                    if not isinstance(req, input_type):
                        raise ValueError("Invalid type: {}".format(type(req)))
                    uid = self._new_uid()
                    fut = asyncio.Future()

                    channel = "{}/{}/{}".format(channel_prefix, "a", uid)
                    self._channel_to_futures[channel] = [fut]
                    self._channel_to_queue[channel] = None

                    self._publish(req, channel_prefix, "q", uid)

                    return fut

                setattr(stub, method_name, functools.partial(method, channel_prefix, input_type))

                self._subscribe(self._client_handler, channel_prefix, "a")
            elif not client_streaming and server_streaming:
                # TODO(nikita): set the name correctly
                def method(channel_prefix, input_type, req):
                    if not isinstance(req, input_type):
                        raise ValueError("Invalid type: {}".format(type(req)))
                    uid = self._new_uid()
                    call = ActiveCall()

                    channel = "{}/{}/{}".format(channel_prefix, "a", uid)
                    self._channel_to_futures[channel] = call.fut_list
                    self._channel_to_queue[channel] = call.queue

                    self._publish(req, channel_prefix, "q", uid)

                    return call

                setattr(stub, method_name, functools.partial(method, channel_prefix, input_type))
                self._subscribe(self._client_handler, channel_prefix, "a")
                self._subscribe(self._client_done_handler, channel_prefix, "A")
            else:
                raise NotImplementedError("Streaming input is not yet implemented")

    def add_to_event_loop(self):
        if self._added_to_event_loop:
            raise Exception("Cannot add LRPC to event loop twice")

        self.loop.add_reader(self.lc.fileno(), self.lc.handle)
        self._added_to_event_loop = True

    def _unpack_channel(self, channel):
        method, code, uid = channel.split('/')
        return code, uid

    def _subscribe(self, handler, name, code, uid=None):
        if uid is None:
            uid = "[A-Za-z0-9]+"
        channel = "{}/{}/{}".format(name, code, uid)
        return self.lc.subscribe(channel, handler)

    def _publish(self, msg, name, code, uid):
        channel = "{}/{}/{}".format(name, code, uid)
        self.lc.publish(channel, msg.SerializeToString())
        return uid

    def _client_handler(self, channel, data):
        if channel not in self._channel_to_queue:
            return

        name = channel.split("/")[0]
        msg = self._name_to_type[name].FromString(data)

        futures = self._channel_to_futures[channel]
        queue = self._channel_to_queue[channel]
        if futures:
            fut = futures.pop(0)
            fut.set_result(msg)
        elif queue is not None:
            queue.append(msg)

        if queue is None and not futures: # Now done!
            del self._channel_to_futures[channel]
            del self._channel_to_queue[channel]

    def _client_done_handler(self, channel, _ignored_data):
        channel = channel.replace('/A/', '/a/')

        if channel not in self._channel_to_queue:
            return

        futures = self._channel_to_futures[channel]

        if futures:
            futures[0].set_result(None)
            for fut in futures[1:]:
                fut.set_exception(Exception("Attempt to read from closed stream"))

        self._channel_to_futures[channel].clear()

        queue = self._channel_to_queue[channel]

        # Now done!
        if queue is not None:
            queue.append(None)

        del self._channel_to_futures[channel]
        del self._channel_to_queue[channel]

    def _new_uid(self):
        return base64.b32encode(uuid.uuid4().bytes[:15]).decode('UTF-8')

# %%

class LCMWrapper(lcm.LCM):
    def publish(self, channel, data):
        print("[LCMWrapper] Publishing to {}".format(channel))
        super().publish(channel, data)

def get_lrpc():
    if LRPC.instance is not None:
        if asyncio.get_event_loop() != LRPC.instance.loop:
            raise Exception("Cannot return LRPC for wrong event loop")

        return LRPC.instance


    loop = asyncio.get_event_loop()
    if loop is None:
        raise Exception("No event loop!")

    import socket
    if socket.gethostname() in ['C02PW77AG8WN', 'fromage']:
        lc = lcm.LCM("udpm://228.6.7.8:7667")
    else:
        lc = lcm.LCM()

    LRPC.instance = LRPC(lc, loop)
    return LRPC.instance

def lcmproto_to_lrpc(lc):
    """
    Helper for transitioning the project from LCMProto to LRPC
    """
    import lcmproto
    LRPC.instance = LRPC(super(lcmproto.LCMProto, lc), asyncio.get_event_loop())
    return LRPC.instance
