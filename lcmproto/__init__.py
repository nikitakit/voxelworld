import lcm
import asyncio
import inspect

class RPCHelper():
    pub_type = None
    sub_type = None
    def __init__(self, lc, publish_channel, subscribe_channel):
        self.lc = lc
        if self.pub_type is None or self.sub_type is None:
            raise NotImplementedError()
        self.publish_channel = self.lc.channel(publish_channel, self.pub_type)
        self.subscription = self.lc.subscribe(subscribe_channel, self.sub_type, self.handler)
        self.digest_to_future = {}

    def handler(self, channel, msg):
        digest = self.digest_sub(msg)
        if digest is None:
            return
        fut = self.digest_to_future.get(digest, None)
        if fut is None:
            return
        self.digest_to_future[digest] = None

        if not fut.done():
            fut.set_result(msg)

    def call(self, msg):
        digest = self.digest_pub(msg)
        if digest is not None:
            fut = asyncio.Future()
            self.digest_to_future[digest] = fut

        self.publish_channel.publish(msg)

        if digest is not None:
            return fut

    def digest_pub(self, msg):
        raise NotImplementedError()

    def digest_sub(self, msg):
        raise NotImplementedError()


class LCMProtoChannel():
    def __init__(self, lc, channel, msg_type):
        self.lc = lc
        self.channel = channel
        self.msg_type = msg_type

    def publish(self, msg):
        if not isinstance(msg, self.msg_type):
            raise ValueError("Invalid message type for publishing")
        self.lc.publish(self.channel, msg)

class LCMProto(lcm.LCM):
    """
    A modified version of LCM that uses protobuf as its message format
    """
    def __init__(self, *args):
        if not args:
            import socket
            if socket.gethostname() in ['C02PW77AG8WN', 'fromage']:
                args = ["udpm://228.6.7.8:7667"]
        super().__init__(*args)

    def channel(self, channel, msg_type):
        return LCMProtoChannel(self, channel, msg_type)

    def publish(self, channel, msg):
        """
        Publishes a protobuf object to a channel
        """
        super().publish(channel, msg.SerializeToString())

    def subscribe(self, channel, msg_type, callback, loop=None):
        def _handle(channel, data):
            nonlocal loop
            res = callback(channel, msg_type.FromString(data))
            if inspect.isawaitable(res):
                if loop is None:
                    loop = asyncio.get_event_loop()
                loop.create_task(res)

        return super().subscribe(channel, _handle)

    def add_to_event_loop(self, loop=None):
        if loop is None:
            loop = asyncio.get_event_loop()
        loop.add_reader(self.fileno(), self.handle)
