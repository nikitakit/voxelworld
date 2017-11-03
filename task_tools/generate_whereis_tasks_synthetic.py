"""
This file is used to generate synthetic Minecraft scenes, place Misty in those
scenes, and then save tasks that will ask workers to annotate those scenes.
"""

# %cd ~/dev/mctest/task_tools
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from util.monkeypatch_voxelregion import monkeypatch
monkeypatch()
import util.struct_tools
from voxelproto.voxelregion_pb2 import VoxelRegion
import math
import random

from learning.data_whereis import RandomRoomData
import hashlib
import numpy as np

from lrpc.lrpc import get_lrpc
import asyncio
from voxelproto import voxelregion_pb2
from voxelproto import world_service_pb2
from voxelproto import crowd_task_pb2
from voxelproto import task_service_pb2
from voxelproto.crowd_task_pb2 import CrowdTask, CrowdTaskResponse
from voxelproto.task_service_pb2 import TaskServiceRequest, NamedTask
from voxelproto.world_service_pb2 import RegionRequest
from voxelproto.world_snapshot_pb2 import WorldSnapshot

TASK_NAME_FORMAT = "/whereis_synth3/{voxels_hash}-{location_num}"
TARGET_LOCATIONS_PER_SNAPSHOT = 1
TARGET_SNAPSHOT_COUNT = 1000
REQUEST_FEEDBACK = False

async def main():
    print("[main] started")
    dg = RandomRoomData(strict=False)
    for _ in range(TARGET_SNAPSHOT_COUNT):
        (voxels, candidates_mask,
            misty_location,
            camera_location, camera_rotation) = dg.example_gen_impl(early_return=True)
        print("Got synthetic scene")

        voxels_hash = hashlib.sha1(voxels).hexdigest()

        candidates_list = []
        for pt in list(zip(*np.nonzero(candidates_mask))):
            candidates_list.extend([int(x) for x in pt])

        task = CrowdTask()
        task.game = "whereis_annotate_v1"
        task.data_static_world.data['x'] = int(misty_location[0])
        task.data_static_world.data['y'] = int(misty_location[1])
        task.data_static_world.data['z'] = int(misty_location[2])
        task.data_static_world.data.get_or_create_list('candidates').extend(candidates_list)

        snapshot = task.data_static_world.snapshot
        snapshot.name = "DUMMY"
        snapshot.position.extend(camera_location)
        snapshot.rotation.extend(camera_rotation)

        region = snapshot.regions.add()
        region.position.extend([0,0,0])
        region.dimensions.extend(list(voxels.shape))

        region.voxels_u32 = np.reshape(np.asarray(voxels, dtype=np.uint32), (-1,))

        # Now we generate multiple possible locations for Misty, and run them
        # by the UI to check that they're sensible
        processed_task = CrowdTask()
        processed_task.MergeFrom(task)
        processed_task.game = "whereis_annotate_create_v1"
        try:
            candidate_locations = list(processed_task.data_static_world.data['candidates'])
        except ValueError:
            print("No candidate locations for Misty!")
            continue
        num_locations = len(candidate_locations) // 3
        print("Got candidate locations:", num_locations)

        good_misty_locations = []
        bad_misty_locations = []

        while (len(good_misty_locations) + len(bad_misty_locations) < num_locations and
               len(good_misty_locations) < TARGET_LOCATIONS_PER_SNAPSHOT):
            num = random.randrange(num_locations)
            while num in good_misty_locations or num in bad_misty_locations:
                num = random.randrange(num_locations)
            processed_task.data_static_world.data['x'] = candidate_locations[num * 3]
            processed_task.data_static_world.data['y'] = candidate_locations[num * 3 + 1]
            processed_task.data_static_world.data['z'] = candidate_locations[num * 3 + 2]

            if REQUEST_FEEDBACK:
                task_response = await tracker_rpc.Activate(processed_task)
                print("Got feedback:", task_response)

                choice = task_response.data_struct['choice']

                if choice == "save":
                    good_misty_locations.append(num)
                elif choice == "reroll":
                    bad_misty_locations.append(num)
                elif choice == "skip":
                    break
            else:
                good_misty_locations.append(num)

        # Now save the resulting tasks
        print("Saving tasks for this synthetic scene:", len(good_misty_locations))
        req = TaskServiceRequest.SubmitTasks()
        task_template = NamedTask()
        task_template.base_task.MergeFrom(task)
        req.tasks.extend([task_template])
        req.save = True
        req.return_processed = True

        data_struct = req.tasks[0].base_task.data_static_world.data
        for i, num in enumerate(good_misty_locations):
            data_struct['x'] = candidate_locations[num * 3]
            data_struct['y'] = candidate_locations[num * 3 + 1]
            data_struct['z'] = candidate_locations[num * 3 + 2]
            req.tasks[0].name = TASK_NAME_FORMAT.format(
                voxels_hash=voxels_hash,
                location_num=i
            )
            print("Saving Misty location {} {} {} to name {}".format(
                data_struct['x'], data_struct['y'], data_struct['z'], req.tasks[0].name
            ))
            await task_rpc.Submit(req)
    print("[main] done")

lrpc = get_lrpc()
tracker_rpc = crowd_task_pb2.TrackerService(lrpc)
task_rpc = task_service_pb2.TaskService(lrpc)

lrpc.add_to_event_loop()
# from util import ipyloop
asyncio.get_event_loop().run_until_complete(main())
