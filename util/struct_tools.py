from google.protobuf.struct_pb2 import Struct
from google.protobuf import json_format
import json

def dict_to_struct(d):
    return json_format.Parse(json.dumps(d), Struct())

def struct_to_dict(s):
    return json.loads(json_format.MessageToJson(s))

def init_from_dict(s, d):
    s.Clear()
    json_format.Parse(json.dumps(d), s)

def msg_to_struct(msg):
    json_msg = json_format.MessageToJson(msg)
    return json_format.Parse(json_msg, Struct())
