import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from voxelproto import voxelregion_pb2
from google.protobuf.internal import python_message
from google.protobuf.internal.encoder import _EncodeVarint, _VarintSize
from google.protobuf.internal.decoder import _VarintDecoder
import six
if six.PY3:
    long = int
_DecodeVarint = _VarintDecoder((1 << 64) - 1, long)
import numpy as np
from copy import copy

# TODO(nikita): this way of fixing the pretty-printer is a really nasty hack
from google.protobuf import text_encoding

old_CEscape = text_encoding.CEscape

def new_CEscape(text, as_utf8):
    if isinstance(text, np.ndarray):
        return "<omitted>"
    return old_CEscape(text, as_utf8)

field_descriptor = copy(voxelregion_pb2.__DUMMYVOXELREGION.fields_by_name['voxels_u32'])
u8_field_descriptor = voxelregion_pb2._VOXELREGION.fields_by_name['voxels_u8']

# Modified encode

def numpy_u32_sizer(value):
    value_len = value.shape[0] * 4 # uint32 is 4 bytes wide
    value_tagged_len = 1 + _VarintSize(value_len) + value_len
    return 1 + _VarintSize(value_tagged_len) + value_tagged_len

def numpy_u32_encoder(write, value):
    value_bytes = value.tobytes()
    value_len = len(value_bytes)
    tag1 = (3 << 3) | 2
    tag2 = (1 << 3) | 2
    write(bytes([tag1]))
    _EncodeVarint(write, 1 + _VarintSize(value_len) + value_len)
    write(bytes([tag2]))
    _EncodeVarint(write, value_len)
    return write(value_bytes)

field_descriptor._sizer = numpy_u32_sizer
field_descriptor._encoder = numpy_u32_encoder

# Modified decode

def DecodeNumpyU32Array(buffer, pos, end, message, field_dict):
    key = field_descriptor

    # Read off varint 1 (length of data used to pack U32Voxels)
    (size, pos) = _DecodeVarint(buffer, pos)
    end_pos = pos + size
    if end_pos > end:
        raise _DecodeError('Truncated string.')

    # Empty array
    if size == 0:
        field_dict[key] = np.array([], dtype=np.uint32)
        return end_pos

    # Read off field name for U32Voxels::voxels_u32
    field_info = buffer[pos]
    pos += 1
    if field_info != ((1 << 3) | 2):
        raise _DecodeError('Invalid field in U32Voxels')

    (size, pos) = _DecodeVarint(buffer, pos)
    end_pos = pos + size
    if pos + size > end:
        raise _DecodeError('Truncated string.')
    if pos + size != end_pos:
        raise _DecodeError('U32Voxels does not have expected encoding.')

    field_dict[key] = np.frombuffer(buffer[pos:end_pos], dtype=np.uint32)
    return end_pos

# Modified accessors

def u32_getter(self):
    field_value = self._fields.get(field_descriptor)
    if field_value is None:
        field_value = np.array([], dtype=np.uint32)
        field_value = self._fields.setdefault(field, field_value)
    return field_value

def u32_setter(self, new_value):
    if not isinstance(new_value, np.ndarray):
        raise ValueError("Can only use numpy array as value")
    if new_value.dtype != np.uint32:
        raise ValueError("dtype must be uint32")
    if len(new_value.shape) != 1:
        raise ValueError("must by 1-d vector")

    self._fields[field_descriptor] = new_value

    # Update Oneof state
    if u8_field_descriptor in self._fields:
        del self._fields[u8_field_descriptor]

    self._Modified()

# Call the function below to actually perform the monkey-patching

HAVE_MONKEYPATCHED = False
def monkeypatch():
    """
    Monkey-patches the VoxelRegion class to store voxels_u32 as a flat numpy
    array instead of a U32Voxels object.

    Serialization/deserialization are supported, but more obscure features may
    not be.
    """
    global HAVE_MONKEYPATCHED
    if HAVE_MONKEYPATCHED:
        return
    HAVE_MONKEYPATCHED = True

    for k, v in voxelregion_pb2._DummyVoxelRegion._decoders_by_tag.items():
        voxelregion_pb2.VoxelRegion._decoders_by_tag[k] = (DecodeNumpyU32Array, None)
    voxelregion_pb2.VoxelRegion.voxels_u32 = property(u32_getter, u32_setter)

    text_encoding.CEscape = new_CEscape
