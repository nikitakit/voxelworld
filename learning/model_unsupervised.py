"""
This is the model we use to train our contextual block embeddings.
"""

try:
    # This file can be run interactively using Hydrogen
    # See: https://github.com/nteract/hydrogen/
    import xdbg
    INTERACTIVE = True
    get_ipython().magic('cd ~/dev/mctest/learning')
except:
    INTERACTIVE = False

#%%

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# %%

import math
import numpy as np
import scipy.ndimage.filters
import tensorflow as tf
import time
import pickle

from data_whereis import DataReader, RandomRoomData
# from data_whereis_networked import SampledWorldData
from util import tf_util

# %%

from matplotlib import pyplot as plt
if INTERACTIVE:
    get_ipython().magic('matplotlib inline')

# %%
gpu_options = tf.GPUOptions()
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# %%
global_step = tf.contrib.framework.get_or_create_global_step()

# %%
batch_size = 10

# dg = SampledWorldData()
dg = RandomRoomData(unsupervised=True)
voxels, example_id = dg.get_inputs(batch_size=batch_size)

coord = tf.train.Coordinator()
coord._my_threads = (list(dg.create_threads(sess, coord, start=True))
                     + list(tf.train.start_queue_runners(sess=sess, coord=coord)))
coord.should_stop()

# %%

known_initialized_variables = set()

def varinit():
    global known_initialized_variables
    to_initialize = set()
    for var in tf.all_variables():
        if var not in known_initialized_variables:
            to_initialize.add(var)
    tf.initialize_variables(list(to_initialize)).run()
    known_initialized_variables |= to_initialize

def varclear():
    global known_initialized_variables
    known_initialized_variables = set(tf.all_variables())
    tf.initialize_all_variables().run()

# %%

NUM_BLOCKS = 256 # dg.max_blocks
block_emb_size = 256

with tf.device("/cpu:0"):
    block_embedding = tf.get_variable("block_embedding_b", [NUM_BLOCKS, block_emb_size],
        trainable=True)

set_block_embedding = tf_util.create_row_setter(block_embedding, "set_block_embedding")

# %%

dynamic_batch_size = tf.shape(voxels)[0]

# %%

layer1_size = 256

with tf.variable_scope("conv1_1"):
    filter_bank = tf.get_variable("filter_bank", [2, 2, 2, block_emb_size, layer1_size], trainable=True)

voxel_emb_1 = tf.nn.conv3d(tf.nn.embedding_lookup(block_embedding, voxels % NUM_BLOCKS),
    filter_bank, strides=[1,1,1,1,1], padding="VALID")

voxel_emb_1s = tf.nn.sigmoid(voxel_emb_1)

# %%
layer2_size = 256

filter2_valid = np.ones((4,4,4), dtype=bool)
filter2_valid[1:3,1:3,1:3] = False
filter2_valid = filter2_valid.reshape((4,4,4,1,1))
filter2_valid = tf.constant(filter2_valid, dtype=tf.float32)

with tf.variable_scope("conv2_2"):
    filter_bank_2 = tf.get_variable("filter_bank", [4, 4, 4, layer1_size, layer2_size], trainable=True)
    filter_bank_2 = filter_bank_2 * filter2_valid

voxel_emb_2 = tf.nn.conv3d(voxel_emb_1, filter_bank_2, strides=[1,1,1,1,1], padding="VALID")

voxel_emb_2s = tf.nn.sigmoid(voxel_emb_2)

# %%

location_emb = tf.reshape(voxel_emb_2s, (-1, layer2_size))

with tf.variable_scope("output_1") as scope:
    class_logits = tf_util.linear(location_emb, 256, bias=False, scope=scope)

# %%

correct_labels = voxels[:,2,2,2] % 256

example_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(class_logits, correct_labels)
loss = tf.reduce_mean(example_losses)

example_errors = tf.not_equal(
    tf.argmax(class_logits, 1),
    correct_labels
)

error = tf.reduce_mean(tf.to_float(example_errors))

# %%

loss_summary = tf.scalar_summary("train/loss", loss)
error_summary = tf.scalar_summary("train/error", error)

valid_loss_summary = tf.scalar_summary("valid/loss", loss, collections=['valid_summaries'])
valid_error_summary = tf.scalar_summary("valid/error", error, collections=['valid_summaries'])

# %%
lr = tf.Variable(0.05, dtype=tf.float32)
lr_summary = tf.scalar_summary("train/lr", lr)

set_lr = tf_util.create_var_setter(lr)
# %%

opt = tf.train.AdamOptimizer(lr)

grads_and_vars = opt.compute_gradients(loss)

apply_gradients_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

with tf.control_dependencies([apply_gradients_op]):
    train_op = tf.no_op(name="train")

# %%
merged_summary = tf.merge_all_summaries()

merged_valid_summary = tf.merge_all_summaries('valid_summaries')

# %%

varinit()

# %%

train_feeds = []
for count in range(10):
    feed = sess.run({
        voxels: voxels,
        example_id: example_id,
    })
    train_feeds.append(feed)

sess.run(loss, train_feeds[0])

# %%

def eval_model(feeds=None):
    if feeds is None:
        feeds = train_feeds
    total_err, total_loss, denom = 0, 0, 0
    for feed_dict in feeds:
        derr, dloss = sess.run([error, loss], feed_dict)
        total_err += derr
        total_loss += dloss
        denom += 1
    return total_err / denom, total_loss / denom

def eval_correct_model(feeds=None):
    if feeds is None:
        feeds = train_feeds
    total_err, total_loss, denom = 0, 0, 0
    for feed_dict in feeds:
        derr, dloss = sess.run([error, loss], feed_dict)
        if derr == 0:
            total_err += derr
            total_loss += dloss
            denom += 1
    return total_err / denom, total_loss / denom

eval_model()

# %%

def inspect_one(feed_dict={}, idx=0, full_range=False, task=None):
    global _last_task

    # This import is not needed for non-interactive mode
    from task_tools import whereis_inspection
    if task is not None:
        whereis_inspection.activate_task(task)
        return

    if task is not None:
        whereis_inspection.activate_task(task)
        return

    vals = tf_util.safe_run(sess, {
        'example_id': example_id,
        'voxels':voxels,
        'error': example_errors,
        'loss': example_losses,
        }, feed_dict)

    for k in vals:
        vals[k] = vals[k][idx]
    print('Error is', vals['error'])

    if isinstance(dg, DataReader):
        assert False, "not supported yet"
        task = whereis_inspection.get_task_from_example_id(vals['example_id']).result()
    else:
        heatmap = np.zeros_like(vals['voxels'], dtype=float)
        heatmap[2,2,2] = 1
        task = whereis_inspection.get_task_from_values(vals['voxels'],
            [
                "Error: {}; Loss: {}".format(vals['error'], vals['loss'])
            ], heatmap, [-1000,-1000,-1000], # Dummy misty location
            vals['example_id'])

    whereis_inspection.activate_task(task, from_values=(not isinstance(dg, DataReader)))

    _last_task = task

# inspect_one()

# %%

def save_filters():
    filter_params = {
        'block_embedding': block_embedding.eval(),
        'filter_bank': filter_bank.eval(),
        'filter_bank_2': filter_bank_2.eval(),
    }
    with open('whereis_filters.pkl', 'wb') as f:
        pickle.dump(filter_params, f)

#%%

# inspect_one(train_feeds[0], idx=3)

# inspect_one()



# %%

set_lr(0.0005)

i = global_step.eval()
while i < 20000:
    if i % 500 == 0:
        print(i, eval_model())

    sess.run(train_op)
    i += 1
else:
    print("Finished training. Saving.")
    save_filters()

# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
