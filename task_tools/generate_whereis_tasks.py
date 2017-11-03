"""
This file is used to take snapshots of Minecraft save files, place Misty in
those snapshots, and then save tasks that will ask workers to annotate those scenes.

(see `generate_whereis_tasks_synthetic.py` for the generation procedure used for
our synthetic scenes)
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
from voxelproto import snapshot_service_pb2

SNAPSHOT_SEARCH = "/whereis2"
TASK_NAME_FORMAT = "/whereis2/{snapshot_basename}-{location_num}"
TARGET_LOCATIONS_PER_SNAPSHOT = 5

async def main():
    print("[main] started")
    req = snapshot_service_pb2.FindSnapshots()
    req.paths.extend([SNAPSHOT_SEARCH])
    response = await snapshot_rpc.Find(req)
    for snapshot in response.snapshots:
        print("Got snapshot")
        print(snapshot)

        snapshot_name = snapshot.name
        while snapshot.name.endswith('/'):
            snapshot.name = snapshot.name[:-1]
        snapshot_basename = snapshot.name.split('/')[-1]

        req = TaskServiceRequest.SubmitTasks()
        task_template = NamedTask()
        task_template.base_task.game = "whereis_annotate_v1"
        task_template.base_task.data_static_world.snapshot.MergeFrom(snapshot)
        req.tasks.extend([task_template])
        req.return_processed = True
        response = await task_rpc.Submit(req)

        # Now we generate multiple possible locations for Misty, and run them
        # by the UI to check that they're sensible
        processed_task = CrowdTask()
        processed_task.MergeFrom(response.tasks[0].processed_task)
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
            task_response = await tracker_rpc.Activate(processed_task)
            print("Got feedback:", task_response)

            choice = task_response.data_struct['choice']

            if choice == "save":
                good_misty_locations.append(num)
            elif choice == "reroll":
                bad_misty_locations.append(num)
            elif choice == "skip":
                break

        # Now save the resulting tasks
        print("Saving tasks for this snapshot:", len(good_misty_locations))
        req = TaskServiceRequest.SubmitTasks()
        task_template = NamedTask()
        task_template.base_task.game = "whereis_annotate_v1"
        task_template.base_task.data_static_world.snapshot.MergeFrom(snapshot)
        task_template.base_task.data_static_world.data.get_or_create_list('candidates').extend(candidate_locations)
        req.tasks.extend([task_template])
        req.save = True
        req.return_processed = True

        data_struct = req.tasks[0].base_task.data_static_world.data
        for i, num in enumerate(good_misty_locations):
            data_struct['x'] = candidate_locations[num * 3]
            data_struct['y'] = candidate_locations[num * 3 + 1]
            data_struct['z'] = candidate_locations[num * 3 + 2]
            req.tasks[0].name = TASK_NAME_FORMAT.format(
                snapshot_basename=snapshot_basename,
                location_num=i
            )
            print("Saving Misty location {} {} {} to name {}".format(
                data_struct['x'], data_struct['y'], data_struct['z'], req.tasks[0].name
            ))
            await task_rpc.Submit(req)
    print("[main] done")

lrpc = get_lrpc()
snapshot_rpc = snapshot_service_pb2.SnapshotService(lrpc)
tracker_rpc = crowd_task_pb2.TrackerService(lrpc)
task_rpc = task_service_pb2.TaskService(lrpc)

lrpc.add_to_event_loop()
# from util import ipyloop
asyncio.get_event_loop().run_until_complete(main())
