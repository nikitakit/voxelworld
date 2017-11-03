# %cd ~/dev/mctest/learning

# import xdbg

# %%
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from lrpc.lrpc import get_lrpc
import asyncio

import tensorflow as tf

from voxelproto.task_service_pb2 import TaskService, TaskServiceRequest, NamedTask
from voxelproto import world_service_pb2
from voxelproto.crowd_task_pb2 import CrowdTask, CrowdTaskResponse

from task_tools.validate_responses import keep_response

import numpy as np
from task_tools.raycast import get_camera_vector
import random

import tornado
from tornado.options import define, options

from nltk import word_tokenize
try:
    word_tokenize('testing the tokenizer')
except LookupError:
    print("NOTE: this script requires nltk tokenizer to be installed")
    raise

# %%

define('path', default=None, multiple=True,
    help='prefix specifying which responses to extract', type=str)

define('out', default='responses.tfrecords',
    help='file to write responses to', type=str)

define('test_num', default=100,
    help='Number of tasks to set aside for a test set', type=int)

define('test_filename', default=None,
    help='File with list of tasks to set aside for the test set', type=str)

define('vocab', default=None,
    help='Vocabulary file to match indices from', type=str)

tornado.options.parse_command_line()

if options.path is None:
    raise ValueError("Need a path option")

# %%

lrpc = get_lrpc()
task_rpc = TaskService(lrpc)
world_rpc = world_service_pb2.WorldService(lrpc)

lrpc.add_to_event_loop()

# %%

def get_example(example_id, words, voxels, candidates_mask):
    # NOTE: no misty_location.
    # Misty is always in the center, and the location randomization occurs in
    # the DataGenerator object
    example = tf.train.Example(features=tf.train.Features(feature={
        'voxels': tf.train.Feature(int64_list=tf.train.Int64List(value=voxels.reshape([-1]).tolist())),
        'words': tf.train.Feature(int64_list=tf.train.Int64List(value=words)),
        'example_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(example_id, 'utf-8')])),
        'candidates_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=candidates_mask.astype(int).reshape([-1]).tolist())),
    }))

    return example

async def write_responses(writer, task_name, base_task, responses, vocabulary):
    camera_vector = get_camera_vector(base_task.data_static_world.snapshot.rotation)

    # Project camera vector onto the horizontal XZ plane
    camera_vector = camera_vector[[0,2]]
    camera_vector /= np.linalg.norm(camera_vector)

    data = base_task.data_static_world.data

    voxels_origin = [int(data['x']) - 18, int(data['y']) - 18, int(data['z']) - 18]

    snapshot = base_task.data_static_world.snapshot
    if not snapshot.regions:
        # TODO: don't require the right world to be pre-loaded by world server
        req = world_service_pb2.RegionRequest()
        candidates_offset = voxels_origin
        req.position.extend(voxels_origin)
        req.dimensions.extend([19+18, 19+18, 19+18])

        resp = await world_rpc.GetRegions(req)
        voxels = np.reshape(resp.regions[0].voxels_u32, resp.regions[0].dimensions)
    elif len(snapshot.regions) > 1:
        assert False # unsupported for now
    elif len(snapshot.regions) == 1:
        # We have the world data in the base task
        # For now, we assume that this is because we're in a synthetic task
        dx, dy, dz = snapshot.regions[0].dimensions
        voxels_position = np.asarray(snapshot.regions[0].position, dtype=int)
        voxels_raw = np.reshape(snapshot.regions[0].voxels_u32, [dx, dy, dz])

        candidates_offset = np.asarray(voxels_origin, dtype=int) + voxels_position

        voxels = np.zeros((19+18, 19+18, 19+18), dtype=voxels_raw.dtype)
        xx, yy, zz = -candidates_offset
        voxels[xx:xx+dx, yy:yy+dy, zz:zz+dz] = voxels_raw


    candidates_mask = np.zeros_like(voxels, dtype=bool)
    candidates = np.array(data['candidates'], dtype=int).reshape((-1, 3))
    candidates -= candidates_offset
    for i in range(candidates.shape[0]):
        try:
            candidates_mask[candidates[i,0], candidates[i,1], candidates[i,2]] = True
        except IndexError:
            # We only use a 19x19x19 window, so some candidates naturally fall
            # outside it
            continue

    for response in responses:
        if not keep_response(response):
            continue
        annotation = response.data_struct["annotation"]
        tokens = word_tokenize(annotation)
        for word in tokens:
            if word not in vocabulary:
                vocabulary.append(word)
        token_idxs = [vocabulary.index(word) for word in tokens]

        example_id = response.assignment_id + "::" + task_name
        example = get_example(example_id, token_idxs, voxels, candidates_mask)
        writer.write(example.SerializeToString())
# %%

async def main(paths, outfile, test_num=0, test_filename=None, vocab_filename=None):
    if not paths:
        print("No paths specified.")
        return

    vocabulary = []
    if vocab_filename is not None:
        with open(vocab_filename, 'r') as f:
            for line in f.readlines():
                vocabulary.append(line[:-1])

    train_excludes = set()
    dev_includes = {}
    test_includes = {}
    includes = test_includes
    if test_filename is not None:
        test_num = 0

        with open(test_filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("# DEV"):
                    includes = dev_includes
                    continue
                elif line.startswith("# TEST"):
                    includes = test_includes
                    continue

                try:
                    guess_name = line.strip()
                    response_idx = int(guess_name.split('-')[-1])
                except:
                    continue

                req = TaskServiceRequest.FindTasks()
                req.names.extend([guess_name])
                req.return_responses = True
                guess_msg = await task_rpc.Find(req)
                assert guess_msg.tasks

                choice_candidates = guess_msg.tasks[0].base_task.data_static_world.data['choice_candidates']
                orig_name = guess_msg.tasks[0].base_task.data_static_world.data['task_name']

                includes[orig_name] = (response_idx, choice_candidates)
                train_excludes.add('-'.join(orig_name.split('-')[:-1]))

        if not train_excludes or (not dev_includes and not test_includes):
            raise Exception("test_filename provided, but no dev/test examples will be set aside")

    req = TaskServiceRequest.FindTaskNames()
    req.paths.extend(paths)
    names_msg = await task_rpc.FindNames(req)
    tasks = []
    tasks_per_chunk = 20
    for i in range(0, len(names_msg.names), tasks_per_chunk):
        req = TaskServiceRequest.FindTasks()
        req.names.extend(names_msg.names[i:i+tasks_per_chunk])
        req.return_responses = True
        msg = await task_rpc.Find(req)
        print('got chunk', i)
        tasks.extend(msg.tasks)

    print("Got full task list")
    writer = tf.python_io.TFRecordWriter(outfile)

    if dev_includes:
        dev_writer = tf.python_io.TFRecordWriter(outfile.replace('.tfrecords', '.dev.tfrecords'))

    if test_num > 0 or test_includes:
        test_writer = tf.python_io.TFRecordWriter(outfile.replace('.tfrecords', '.test.tfrecords'))

    for named_task in tasks:
        responses = named_task.responses
        if not responses:
            continue
        if named_task.name in dev_includes or named_task.name in test_includes:
            if named_task.name in dev_includes:
                print('Setting aside for dev set:', named_task.name)
                use_writer = dev_writer
                response_idx, choice_candidates = dev_includes[named_task.name]
            else:
                print('Setting aside for test set:', named_task.name)
                use_writer = test_writer
                response_idx, choice_candidates = test_includes[named_task.name]

            responses = [responses[response_idx]]
            named_task.base_task.data_static_world.data['candidates'].Clear()
            named_task.base_task.data_static_world.data['candidates'].extend(choice_candidates)
            assert keep_response(responses[0])
        elif '-'.join(named_task.name.split('-')[:-1]) in train_excludes:
            print('Not including in training set:', named_task.name)
            continue
        elif test_num > 0:
            print('Setting aside for test set:', named_task.name)
            use_writer = test_writer
            test_num -= 1
        else:
            use_writer = writer
        await write_responses(use_writer, named_task.name, named_task.base_task, responses, vocabulary)

    vocab_file = outfile.replace('tfrecords', 'vocab')
    with open(vocab_file, 'w') as f:
        for word in vocabulary:
            f.write(word + '\n')

# %%
# from util import ipyloop

# %%

# asyncio.ensure_future(main('whereis2', 'responses.tfrecords'))

# %%

asyncio.get_event_loop().run_until_complete(main(options.path, options.out, options.test_num, options.test_filename, options.vocab))
