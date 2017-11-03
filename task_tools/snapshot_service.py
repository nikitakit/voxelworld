# %cd ~/dev/mctest/task_tools

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from util.monkeypatch_voxelregion import monkeypatch
monkeypatch()
from voxelproto.voxelregion_pb2 import VoxelRegion
import math

from lrpc.lrpc import get_lrpc
import asyncio
from voxelproto import voxelregion_pb2
from voxelproto import world_service_pb2
from voxelproto import snapshot_service_pb2
from voxelproto.world_snapshot_pb2 import WorldSnapshot

SNAPSHOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
    'snapshots'
)
SNAPSHOTS = {}

def init_snapshots_from_disk():
    print("Loading snapshots...")
    for folder, subdirs, files in os.walk(SNAPSHOT_DIR):
        for filename in files:
            if not filename.endswith('.pb'):
                continue
            with open(os.path.join(folder, filename), 'rb') as f:
                snapshot = WorldSnapshot.FromString(f.read())
                print('-', snapshot.name)
                SNAPSHOTS[snapshot.name] = snapshot
                snapshot.ClearField('regions')

def save_snapshot_to_disk(snapshot):
    folder = os.path.join(SNAPSHOT_DIR, os.path.dirname(snapshot.name))
    filename = os.path.join(folder, os.path.basename(snapshot.name) + '.pb')
    print('Saving snapshot to:', filename)

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(filename, 'wb') as f:
        f.write(snapshot.SerializeToString())

def normalize_path(path):
    while path.startswith('/'):
        path = path[1:]
    while '//' in path:
        path = path.replace('//', '/')
    return path

async def save_snapshots(msg):
    for snapshot in msg.snapshots:
        if len(snapshot.regions) != 0:
            return

        snapshot.name = normalize_path(snapshot.name)

        SNAPSHOTS[snapshot.name] = snapshot
        save_snapshot_to_disk(snapshot)

class SnapshotServiceServicerImpl(snapshot_service_pb2.SnapshotServiceServicer):
    async def ListFolders(self, msg):
        folders = set()
        for key in SNAPSHOTS:
          folder = '/'.join(key.split('/')[:-1])
          folder = normalize_path(folder)

          folders.add(folder)

        response = snapshot_service_pb2.ListFoldersResponse()
        response.folders.extend(sorted(folders))
        return response

    async def Find(self, msg):
        response = snapshot_service_pb2.FindSnapshotsResponse()

        for path in msg.paths:
            path = normalize_path(path)

            snapshots = [val for (key, val) in SNAPSHOTS.items() if key.startswith(path)]
            response.snapshots.extend(sorted(snapshots, key=lambda snapshot: snapshot.name))

        for name in msg.names:
            name = normalize_path(name)
            if name in SNAPSHOTS:
                response.snapshots.extend([SNAPSHOTS[name]])

        return response

    async def Save(self, msg):
        await save_snapshots(msg)
        return snapshot_service_pb2.SaveSnapshotsResponse()

lrpc = get_lrpc()
lrpc.add_servicer(SnapshotServiceServicerImpl())
lrpc.add_to_event_loop()


init_snapshots_from_disk()
#from util import ipyloop
asyncio.get_event_loop().run_forever()
