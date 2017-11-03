"""
This file help inspect the results of tasks in the graphical console.
It can be run as a script, or imported from other files
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
import time
import numpy as np

from threading import Thread

from lrpc.lrpc import get_lrpc
import asyncio
from voxelproto import voxelregion_pb2
from voxelproto import world_service_pb2
from voxelproto import crowd_task_pb2
from voxelproto import task_service_pb2
from voxelproto.crowd_task_pb2 import CrowdTask, CrowdTaskResponse, Heatmap
from voxelproto.task_service_pb2 import NamedTask
from voxelproto.world_service_pb2 import RegionRequest

loop = asyncio.new_event_loop()

def wrap_threadsafe(fn):
    def wrapped_fn(*args, **kwargs):
        global loop
        return asyncio.run_coroutine_threadsafe(fn(*args, **kwargs), loop)
    return wrapped_fn

@wrap_threadsafe
async def get_all_task_responses(path):
    req = task_service_pb2.TaskServiceRequest.FindTasks()
    req.paths.extend([path])
    req.return_responses = True
    response = await task_rpc.Find(req)

    res = []
    for task in response.tasks:
        for task_response in task.responses:
            if task_response.completed == 1 and 'test' not in task_response.assignment_id and len(task_response.assignment_id) != 11:
                if len(task_response.data_struct['pose_history']) <= 7:
                    continue
                new_task = NamedTask()
                new_task.name = task.name
                new_task.base_task.MergeFrom(task.base_task)
                new_task.responses.extend([task_response])
                res.append(new_task)

    return res

async def _get_individual_task_responses(name):
    req = task_service_pb2.TaskServiceRequest.FindTasks()
    req.names.extend([name])
    req.return_responses = True
    response = await task_rpc.Find(req)
    if not response.tasks:
        return None
    return response.tasks[0]
get_individual_task_responses = wrap_threadsafe(_get_individual_task_responses)

@wrap_threadsafe
async def get_task_from_example_id(example_id):
    if not isinstance(example_id, str):
        example_id = example_id.decode('utf-8')
    assignment_id, task_name = example_id.split('::', 1)
    named_task = await _get_individual_task_responses(task_name)
    while named_task.responses and named_task.responses[0].assignment_id != assignment_id:
        del named_task.responses[0]
    del named_task.responses[1:]
    if not named_task.responses:
        return None
    return named_task

def get_task_from_values(voxels, words_str, candidates_mask, misty_location, example_id):
    task = NamedTask()
    task.base_task.game = "whereis_review_v1"
    task.responses.add().data_struct['annotation'] = ' '.join(words_str)
    response = task.base_task.data_static_world.data.get_or_create_struct('response')
    response.MergeFrom(util.struct_tools.msg_to_struct(task.responses[0]))

    task.base_task.data_static_world.data['x'] = int(misty_location[0])
    task.base_task.data_static_world.data['y'] = int(misty_location[1])
    task.base_task.data_static_world.data['z'] = int(misty_location[2])

    task.base_task.data_static_world.data.get_or_create_list('candidates').extend(
        np.transpose(np.nonzero(candidates_mask)).flatten().tolist())


    camera_information = [float(x) for x in example_id.decode().split(":")[1:]]

    snapshot = task.base_task.data_static_world.snapshot
    snapshot.name = "DUMMY"
    snapshot.position.extend(camera_information[:3])
    snapshot.rotation.extend(camera_information[3:])

    region = snapshot.regions.add()
    region.position.extend([0,0,0])
    region.dimensions.extend(list(voxels.shape))

    region.voxels_u32 = np.reshape(np.asarray(voxels, dtype=np.uint32), (-1,))

    return task

def prepare_submit_request(task):
    req = task_service_pb2.TaskServiceRequest.SubmitTasks()
    req.tasks.extend([NamedTask()])
    named_task = req.tasks[0]
    named_task.MergeFrom(task)

    named_task.processed_task.Clear()
    named_task.base_task.game = "whereis_review_v1"
    response = named_task.base_task.data_static_world.data.get_or_create_struct('response')
    response.MergeFrom(util.struct_tools.msg_to_struct(named_task.responses[0]))
    del named_task.responses[:]

    return req

def add_heatmap(task, name, origin, voxel_data):
    heatmap = Heatmap()
    heatmap.name = name
    heatmap.position.extend([int(origin[0]), int(origin[1]), int(origin[2])])
    heatmap.dimensions.extend([voxel_data.shape[0], voxel_data.shape[1], voxel_data.shape[2]])

    values = np.asarray(voxel_data, dtype=float)
    if np.min(values) < 0.0 or np.max(values) > 1.0:
        raise ValueError("Heatmap only allows values in the range 0.0-1.0")

    values = np.reshape(values, [-1])
    heatmap.voxels.extend(values.tolist())

    task.base_task.data_static_world.heatmaps.extend([heatmap])

def add_misty_relative_heatmap(task, name, voxel_data, offset=None):
    """
    Add a heatmap to a task, where the data is assumed to have odd side lengths
    and be centered on Misty.
    """
    task_data = task.base_task.data_static_world.data
    origin = np.array([task_data['x'], task_data['y'], task_data['z']])
    if offset is None:
        origin -= [voxel_data.shape[0] // 2, voxel_data.shape[1] // 2, voxel_data.shape[2] // 2]
    else:
        origin -= offset
    return add_heatmap(task, name, origin, voxel_data)


@wrap_threadsafe
async def process_task(task):
    req = prepare_submit_request(task)
    req.return_processed = True
    processed_task_resp = await task_rpc.Submit(req)
    res_task = processed_task_resp.tasks[0]
    del res_task.responses[:]
    res_task.responses.MergeFrom(task.responses)
    return res_task

@wrap_threadsafe
async def activate_task(task, wait=False, from_values=False):
    if from_values:
        a = tracker_rpc.Activate(task.base_task)

        if wait:
            return await a
        else:
            return True

    req = prepare_submit_request(task)

    if wait:
        req.return_processed = True
        processed_task_resp = await task_rpc.Submit(req)
        return await tracker_rpc.Activate(processed_task_resp.tasks[0].processed_task)
    else:
        req.activate = True
        task_rpc.Submit(req)
        return True

tracker_rpc = None
task_rpc = None

def run_event_loop(loop):
    global tracker_rpc, task_rpc
    asyncio.set_event_loop(loop)

    lrpc = get_lrpc()
    tracker_rpc = crowd_task_pb2.TrackerService(lrpc)
    task_rpc = task_service_pb2.TaskService(lrpc)

    lrpc.add_to_event_loop()
    loop.run_forever()

thread = Thread(target=run_event_loop, args=(loop,), daemon=True)
thread.start()

def add_test_heatmap(task):
    origin = task.base_task.data_static_world.snapshot.position
    origin = [int(origin[0]) - 10, int(origin[1]) - 10, int(origin[2]) - 10]

    size = 20 ** 3
    values = np.arange(size, dtype=float) / size
    values = np.reshape(values, (20, 20, 20))
    add_heatmap(task, "test", origin, values)

def test1():
    task_name = "whereis2/2016-10-28T20:23:38-0"
    task = get_individual_task_responses(task_name).result()
    add_test_heatmap(task)
    activated = activate_task(task).result()

def main():
    all_tasks = get_all_task_responses('whereis_synth3').result()
    for task in all_tasks:
        activated = activate_task(task, wait=True).result()
        print('got result', activated)

if __name__ == '__main__':
    main()
