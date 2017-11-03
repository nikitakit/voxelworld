import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# %%

import lcmproto
import asyncio
from voxelproto import voxelregion_pb2
from voxelproto.voxelregion_pb2 import VoxelRegion
from voxelproto.crowd_task_pb2 import CrowdTask

from mceditlib.worldeditor import WorldEditor

# Prefer to work with numpy arrays
import numpy as np
from util.monkeypatch_voxelregion import monkeypatch
monkeypatch()

# %% Open up the editor

WORLD_PATH = "/Users/kitaev/software/malmo/build/install/Minecraft/run/saves/template"

editor = WorldEditor(WORLD_PATH, resume=False)
dimension = editor.getDimension()

# %% Clear all chunks

def handle_task(channel, task):
    print("Received message on channel \"%s\"" % channel)

    snapshot = task.data_static_world.snapshot

    editor.getWorldMetadata().LevelName = task.game
    editor.getWorldMetadata().GameType = 1 # creative
    editor.getPlayer().Position = (snapshot.position[0], snapshot.position[1] - 1.5, snapshot.position[2])
    editor.getPlayer().abilities.invulnerable = 1
    editor.getPlayer().abilities.flying = 1
    editor.getPlayer().abilities.mayfly = 1

    for x, y in list(dimension.chunkPositions()):
        dimension.deleteChunk(x, y)

    for x in range(-5, 5):
        for y in range(-5, 5):
            dimension.createChunk(x, y)

    for region in snapshot.regions:
        voxels = region.voxels_u32.reshape(region.dimensions)
        ox, oy, oz = region.position
        for xx in range(voxels.shape[0]):
            x = ox + xx
            for yy in range(voxels.shape[1]):
                y = oy + yy
                for zz in range(voxels.shape[2]):
                    data = voxels[xx, yy, zz]
                    block_id = data % 256
                    block_meta = data >> 8

                    z = oz + zz
                    dimension.setBlock(x, y, z, (block_id, block_meta))


    data_struct = task.data_static_world.data
    if 'x' in data_struct.fields and 'y' in data_struct.fields and 'z' in data_struct.fields:
        dimension.setBlock(int(data_struct['x']),
            int(data_struct['y']),
            int(data_struct['z']),
            (122, 0)) # Currently Misty replaces the dragon egg, block id 122

    editor.saveChanges()

    print('Done saving world')

lc = lcmproto.LCMProto()
subscription = lc.subscribe("TrackerService.Activate/q/[A-Za-z0-9]+", CrowdTask, handle_task)

lc.add_to_event_loop()
asyncio.get_event_loop().run_forever()
