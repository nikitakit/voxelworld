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
from voxelproto.crowd_task_pb2 import CrowdTask
from voxelproto.task_service_pb2 import TaskServiceRequest, NamedTask
from voxelproto import world_service_pb2
from voxelproto.world_snapshot_pb2 import WorldSnapshot

import watchdog.observers
import watchdog.events

import raycast

EXTENTS = [
    # -x, +x, -z, +z
    [-32, +32, -64, +32],
    [-32, +64, -64, +32],
    [-32, +64, -32, +32],
    [-32, +64, -32, +64],
    [-32, +32, -32, +64],
    [-64, +32, -32, +64],
    [-64, +32, -32, +32],
    [-64, +32, -64, +32],
]


def index_for_angle(angle):
    angle_in_sixteenths = angle / (2 * math.pi / 16)
    angle_in_sixteenths = (1600 + angle_in_sixteenths) % 16
    if -1 <= angle_in_sixteenths <= 1:
        return 0
    elif 1 <= angle_in_sixteenths <= 3:
        return 1
    elif 3 <= angle_in_sixteenths <= 5:
        return 2
    elif 5 <= angle_in_sixteenths <= 7:
        return 3
    elif 7 <= angle_in_sixteenths <= 9:
        return 4
    elif 9 <= angle_in_sixteenths <= 11:
        return 5
    elif 11 <= angle_in_sixteenths <= 13:
        return 6
    elif 13 <= angle_in_sixteenths <= 15:
        return 7
    elif 15 <= angle_in_sixteenths <= 17:
        return 0

def get_region(position, rotation):
    rotation_y = rotation[1]
    i = index_for_angle(rotation_y)
    extents_x = EXTENTS[i][:2]
    extents_y = [-16, +16]
    extents_z = EXTENTS[i][2:]

    region_position = [
        int(position[0]) + extents_x[0],
        int(position[1]) + extents_y[0],
        int(position[2]) + extents_z[0],
    ]

    region_dimensions = [
        extents_x[1] - extents_x[0],
        extents_y[1] - extents_y[0],
        extents_z[1] - extents_z[0]
    ]

    res = world_service_pb2.RegionRequest()
    res.position.extend(region_position)
    res.dimensions.extend(region_dimensions)
    return res

TASK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
    'tasks'
)

RESPONSES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
    'task_responses'
)
PROCESSED_TASKS = {}
UNPROCESSED_TASKS = {}
RESPONSES = {}

def init_tasks_from_disk():
    print("Loading tasks...")
    for folder, subdirs, files in os.walk(TASK_DIR):
        for filename in files:
            if not filename.endswith('.pb'):
                continue
            with open(os.path.join(folder, filename), 'rb') as f:
                task = NamedTask.FromString(f.read())
                unprocessed_task = NamedTask()
                unprocessed_task.CopyFrom(task)
                unprocessed_task.ClearField('processed_task')
                print('-', task.name)
                PROCESSED_TASKS[task.name] = task
                UNPROCESSED_TASKS[unprocessed_task.name] = unprocessed_task

def init_task_responses_from_disk():
    print("Loading task responses...")
    for folder, subdirs, files in os.walk(RESPONSES_DIR):
        for filename in files:
            if not filename.endswith('.pb'):
                continue
            with open(os.path.join(folder, filename), 'rb') as f:
                responses = NamedTask.FromString(f.read())
                print('-', responses.name)
                RESPONSES[responses.name] = responses

def save_task_to_disk(task):
    folder = os.path.join(TASK_DIR, os.path.dirname(task.name))
    filename = os.path.join(folder, os.path.basename(task.name) + '.pb')
    print('Saving task to:', filename)

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(filename, 'wb') as f:
        f.write(task.SerializeToString())

def normalize_path(path):
    while path.startswith('/'):
        path = path[1:]
    while '//' in path:
        path = path.replace('//', '/')
    return path

def list_folders():
    folders = set()
    for key in PROCESSED_TASKS:
        folder = '/'.join(key.split('/')[:-1])
        folder = normalize_path(folder)

        folders.add(folder)

    response = task_service_pb2.ListFoldersResponse()
    response.folders.extend(sorted(folders))
    return response

def find_tasks(msg):
    response = task_service_pb2.FindTasksResponse()

    task_dict = PROCESSED_TASKS if msg.return_processed else UNPROCESSED_TASKS

    for path in msg.paths:
        path = normalize_path(path)

        tasks = [val for (key, val) in task_dict.items() if key.startswith(path)]
        response.tasks.extend(sorted(tasks, key=lambda task: task.name))

    for name in msg.names:
        name = normalize_path(name)
        if name in task_dict:
            response.tasks.extend([task_dict[name]])

    # Note that extend copies the data structures over, so mutating here won't
    # affect the original
    if msg.return_responses:
        for task in response.tasks:
            if task.name in RESPONSES:
                task.responses.MergeFrom(RESPONSES[task.name].responses)

    return response

def find_task_names(msg):
    response = task_service_pb2.FindTaskNamesResponse()

    response_names = set()

    for path in msg.paths:
        path = normalize_path(path)

        names = [key for key in UNPROCESSED_TASKS if key.startswith(path)]
        response_names |= set(names)

    for name in msg.names:
        name = normalize_path(name)
        if name in UNPROCESSED_TASKS:
            response_names.add(name)

    response.names.extend(sorted(response_names))

    return response

async def process_task(raw_task):
    raw_task.processed_task.CopyFrom(raw_task.base_task)

    if raw_task.processed_task.game in ["snapshot_annotate_v3", "whereis_annotate_v1", "whereis_guess_v1", "whereis_review_v1"]:
        snapshot = raw_task.processed_task.data_static_world.snapshot
        if not snapshot.regions:
            response = await world_rpc.GetRegions(get_region(snapshot.position, snapshot.rotation))

            snapshot.regions.MergeFrom(response.regions)

    if raw_task.processed_task.game in ["whereis_annotate_v1", "whereis_review_v1"]:
        data_struct = raw_task.processed_task.data_static_world.data
        if 'candidates' not in data_struct.fields:
            candidates_field = data_struct.get_or_create_list('candidates')
            candidate_locations = raycast.misty_locations_from_snapshot(raw_task.processed_task.data_static_world.snapshot)
            candidates_field.extend(candidate_locations)
        else:
            candidate_locations = list(data_struct['candidates'])

        if 'x' not in data_struct.fields or 'y' not in data_struct.fields or 'z' not in data_struct.fields:
            if len(candidate_locations) == 0:
                print("WARNING: no possible locations for Misty")
            else:
                num = random.randrange(len(candidate_locations) // 3)
                data_struct['x'] = candidate_locations[num * 3]
                data_struct['y'] = candidate_locations[num * 3 + 1]
                data_struct['z'] = candidate_locations[num * 3 + 2]

async def submit_tasks(msg):
    if not msg.save and not msg.return_processed and not msg.activate:
        # Nothing to do , so don't bother processing tasks
        return

    response = task_service_pb2.SubmitTasksResponse()
    task_to_activate = None
    for task in msg.tasks:
        if msg.save:
            unprocessed_task = NamedTask()
            unprocessed_task.CopyFrom(task)
            unprocessed_task.ClearField('processed_task')

        await process_task(task)
        task.name = normalize_path(task.name)

        if msg.save:
            unprocessed_task.name = task.name
            PROCESSED_TASKS[task.name] = task
            UNPROCESSED_TASKS[task.name] = unprocessed_task
            save_task_to_disk(task)

        if msg.return_processed:
            response.tasks.extend([task])

        if msg.activate:
            task_to_activate = task.processed_task

    if task_to_activate is not None:
        tracker_rpc.Activate(task_to_activate)

    return response

class TaskServiceServicerImpl(task_service_pb2.TaskServiceServicer):
    async def ListFolders(self, msg):
        print('ListFolders called')
        return list_folders()

    async def Find(self, msg):
        print('Find called')
        return find_tasks(msg)

    async def Submit(self, msg):
        print('Submit called')
        return await submit_tasks(msg)

    async def FindNames(self, msg):
        print('FindNames called')
        return find_task_names(msg)

HAVE_MODIFIED_RESPONSES = False
class TaskResponsesModifiedHandler(watchdog.events.FileSystemEventHandler):
    def on_any_event(self, info):
        global HAVE_MODIFIED_RESPONSES
        HAVE_MODIFIED_RESPONSES = True

observer = watchdog.observers.Observer()
observer.schedule(TaskResponsesModifiedHandler(), RESPONSES_DIR, recursive=True)
observer.start()

async def poll_reload_responses():
    global HAVE_MODIFIED_RESPONSES
    while True:
        await asyncio.sleep(1.0)
        if HAVE_MODIFIED_RESPONSES:
            print('Detected that responses on disk have been modified')
            HAVE_MODIFIED_RESPONSES = False
            init_task_responses_from_disk()

lrpc = get_lrpc()
world_rpc = world_service_pb2.WorldService(lrpc)
tracker_rpc = crowd_task_pb2.TrackerService(lrpc)
lrpc.add_servicer(TaskServiceServicerImpl())


lrpc.add_to_event_loop()
init_tasks_from_disk()
init_task_responses_from_disk()
#from util import ipyloop

task = asyncio.Task(poll_reload_responses())
asyncio.get_event_loop().run_until_complete(task)
