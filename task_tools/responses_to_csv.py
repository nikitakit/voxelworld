import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from lrpc.lrpc import get_lrpc
import asyncio

import pandas as pd
from voxelproto import task_service_pb2
from voxelproto.task_service_pb2 import NamedTask

from task_tools.validate_responses import keep_response

from google.protobuf import json_format
import json

from datetime import datetime, timezone
import pytz

import tornado
from tornado.options import define, options

define('path', default=None, multiple=True,
    help='prefix specifying which responses to extract', type=str)

define('out', default='responses.csv',
    help='file to write responses to', type=str)

define('filter', default=True,
    help='filter out responses from dev portal testing', type=bool)

tornado.options.parse_command_line()

NUM_RESPONSES=3

if options.path is None:
    raise ValueError("Need a path option")

lrpc = get_lrpc()
task_rpc = task_service_pb2.TaskService(lrpc)

lrpc.add_to_event_loop()
# from util import ipyloop

def get_timestamp(crowd_task_response):
    if not crowd_task_response.HasField('submission_timestamp'):
        return ""
    timestamp = crowd_task_response.submission_timestamp.ToDatetime()
    # Protobuf returns a naive datetime, so we need to manually attach the UTC
    # label
    timestamp = pytz.timezone('UTC').localize(timestamp)
    return timestamp.astimezone(pytz.timezone('America/Los_Angeles')).isoformat()

def named_tasks_to_df(named_tasks, filter=True):
    column_order = ['name', 'completed', 'worker_id', 'assignment_id', 'submission_timestamp', 'elapsed_time']
    res = []
    for named_task in named_tasks:
        name = named_task.name

        if named_task.base_task.game == 'whereis_guess_v2':
            data = named_task.base_task.data_static_world.data
            correct_choice = None
            for i in range(len(data['choice_candidates'])//2):
                if tuple(list(data['choice_candidates'])[i*3:i*3+3]) == (
                    int(data['x']), int(data['y']), int(data['z'])):
                    correct_choice = i
                    break
            assert correct_choice is not None

            count_responses = 0
            workers = set() # make no worker solves the same task twice
            for response in named_task.responses:
                if filter:
                    if not keep_response(response):
                        continue
                    if response.worker_id in workers:
                        continue
                    workers.add(response.worker_id)
                count_responses += 1

            if count_responses < NUM_RESPONSES:
                print("WARNING: too few responses {}. skipping".format(count_responses))
                print(name)
                continue

            count_responses = NUM_RESPONSES
        else:
            count_responses = 1000

        workers = set() # make no worker solves the same task twice
        for response in named_task.responses:
            if filter:
                if not keep_response(response):
                    continue
            if response.worker_id in workers:
                continue
            workers.add(response.worker_id)
            count_responses -= 1
            if count_responses < 0:
                break
            entry = {
                'name': name,
                'assignment_id': response.assignment_id,
                'worker_id': response.worker_id,
                'completed': response.completed,
                'submission_timestamp': get_timestamp(response),
                'elapsed_time': response.elapsed_time
            }

            if named_task.base_task.game == 'whereis_guess_v2':
                if 'correct_choice' not in column_order:
                    column_order.append('correct_choice')
                entry['correct_choice'] = correct_choice

            for k, v in response.data_struct.fields.items():
                entry[k] = str(json.loads(json_format.MessageToJson(v)))
                if k not in column_order:
                    column_order.append(k)
            res.append(entry)
    if 'pose_history' in column_order:
        column_order.remove('pose_history')
    res = pd.DataFrame(res, columns=column_order)
    res = res.sort_values(['completed', 'name'])
    return res

async def main(paths, outfile=None, filter=True):
    if not paths:
        print("No paths specified.")
        return

    req = task_service_pb2.TaskServiceRequest.FindTaskNames()
    req.paths.extend(paths)
    names_msg = await task_rpc.FindNames(req)
    tasks = []
    tasks_per_chunk = 20
    for i in range(0, len(names_msg.names), tasks_per_chunk):
        req = task_service_pb2.TaskServiceRequest.FindTasks()
        req.names.extend(names_msg.names[i:i+tasks_per_chunk])
        req.return_responses = True
        msg = await task_rpc.Find(req)
        print('got chunk', i)
        tasks.extend(msg.tasks)

    df = named_tasks_to_df(tasks, filter=filter)
    df.to_csv(outfile)

# asyncio.ensure_future(main('internal1'))

asyncio.get_event_loop().run_until_complete(main(options.path, options.out, options.filter))
