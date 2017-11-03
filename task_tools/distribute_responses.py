"""
This file prepares responses for distribution.

It will filter out tasks that don't have any human responses, and substitute
worker/assignment IDs for increased anonymity.

For the evaluation task (which involves guessing out of 6 distractors), it will
retain the human responses that were used to evaluate human performance. (It
will not retain the human responses used to filter the dataset in the first
place)
"""

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from lrpc.lrpc import get_lrpc
import asyncio

from voxelproto import task_service_pb2
from voxelproto.task_service_pb2 import NamedTask

from task_tools.validate_responses import keep_response

import tornado
from tornado.options import define, options

define('path', default=None, multiple=True,
    help='prefix specifying which responses to extract', type=str)

define('test_filename', default=None,
    help='File with list of tasks to set aside for the test set (and dev set)', type=str)

tornado.options.parse_command_line()

if options.path is None:
    raise ValueError("Need a path option")

lrpc = get_lrpc()
task_rpc = task_service_pb2.TaskService(lrpc)

lrpc.add_to_event_loop()
# from util import ipyloop

DISTRIB_DIR = os.path.join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
    'distrib/'
)

TASK_DIR = os.path.join(DISTRIB_DIR, 'tasks')
RESPONSES_DIR = os.path.join(DISTRIB_DIR, 'task_responses')

def counter(prefix):
    i = 0
    while True:
        yield "{}_{}".format(prefix, i)
        i += 1

def distrib_tasks(named_tasks):
    # For increased anonymity, replace worker ids and assignment ids with counters
    worker_map = {}

    worker_names = counter('worker')
    assignment_names = counter('assignment')

    for named_task in named_tasks:
        name = named_task.name

        task_copy = NamedTask()
        task_copy.name = named_task.name
        task_copy.base_task.MergeFrom(named_task.base_task)
        task_copy.processed_task.MergeFrom(named_task.processed_task)

        responses_copy = NamedTask()
        responses_copy.name = named_task.name

        workers = set() # make sure no worker solves the same task twice
        count = 0
        for response in named_task.responses:
            if not keep_response(response):
                continue
            if response.worker_id in workers:
                continue
            count += 1
            if named_task.base_task.game == 'whereis_guess_v2' and count != 3:
                # First two responses were used to construct the dev/test set
                # list. Keep only the third, which is used to evaluate human
                # performance.
                continue

            workers.add(response.worker_id)

            if response.worker_id in worker_map:
                anon_worker = worker_map[response.worker_id]
            else:
                anon_worker = next(worker_names)
                worker_map[response.worker_id] = anon_worker

            new_response = responses_copy.responses.add()
            new_response.MergeFrom(response)
            new_response.assignment_id = next(assignment_names)
            new_response.worker_id = anon_worker

        if not responses_copy.responses:
            continue

        # Save the task
        folder = os.path.join(TASK_DIR, os.path.dirname(task_copy.name))
        filename = os.path.join(folder, os.path.basename(task_copy.name) + '.pb')
        print('Saving task to:', filename)

        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(filename, 'wb') as f:
            f.write(task_copy.SerializeToString())

        # Save the responses
        folder = os.path.join(RESPONSES_DIR, os.path.dirname(responses_copy.name))
        filename = os.path.join(folder, os.path.basename(responses_copy.name) + '.pb')
        print('Saving responses to:', filename)

        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(filename, 'wb') as f:
            f.write(responses_copy.SerializeToString())

    print('Number of workers:', len(worker_map))

async def main(paths, test_filename):
    if not paths:
        print("No paths specified.")
        return

    if not test_filename:
        print("No test_filename specified.")
        return

    guess_names = []
    with open(test_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("# DEV") or line.startswith("# TEST"):
                continue

            guess_names.append(line)

    req = task_service_pb2.TaskServiceRequest.FindTaskNames()
    req.paths.extend(paths)
    names_msg = await task_rpc.FindNames(req)
    names = list(names_msg.names) + guess_names

    tasks = []
    tasks_per_chunk = 20
    for i in range(0, len(names), tasks_per_chunk):
        req = task_service_pb2.TaskServiceRequest.FindTasks()
        req.names.extend(names[i:i+tasks_per_chunk])
        req.return_responses = True
        msg = await task_rpc.Find(req)
        print('got chunk', i)
        tasks.extend(msg.tasks)

    distrib_tasks(tasks)

asyncio.get_event_loop().run_until_complete(main(options.path, options.test_filename))
