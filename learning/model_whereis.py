"""
This is the main model for the "Where is Misty?" task
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
import glob

from data_whereis import DataReader, RandomRoomData
from util import tf_util

# %%

from matplotlib import pyplot as plt

if INTERACTIVE:
    get_ipython().magic('matplotlib inline')

# %%

ABLATION = None
assert ABLATION in [None, 'single_block', 'constant_filter', 'no_lstm']

SAVER_DIR = os.path.expanduser('~/tf_logs/{}'.format('whereis_d_testdropout_3'))
EVALUATE = False

if not os.path.exists(SAVER_DIR):
    assert not EVALUATE
    os.makedirs(SAVER_DIR)

# %%
gpu_options = tf.GPUOptions()
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# %%
global_step = tf.contrib.framework.get_or_create_global_step()

# %%
batch_size = 10

dg = DataReader("whereis.tfrecords")
voxels, words, candidates_mask, misty_location, example_id = dg.get_inputs(batch_size=batch_size)
is_training = tf.placeholder_with_default(True, shape=())

coord = tf.train.Coordinator()
coord._my_threads = (list(dg.create_threads(sess, coord, start=True))
                     + list(tf.train.start_queue_runners(sess=sess, coord=coord)))
coord.should_stop()

dev_feeds = dg.get_dev_feeds()

for fd in dev_feeds:
    fd[is_training] = False

test_feeds = dg.get_test_feeds()

for fd in test_feeds:
    fd[is_training] = False

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

VOCAB, EMBEDDINGS = dg.get_vocab_embeddings()
word_emb_size = EMBEDDINGS.shape[1]
VOXEL_VALS_SIZE = 19

# %%

batch_norm_ema = tf.train.ExponentialMovingAverage(decay=0.99)
step_ops = {}

def batch_norm(variable, axes):
    beta_gamma_shape = [x for i, x in enumerate(variable.get_shape()) if i not in axes]
    beta = tf.Variable(tf.constant(0.0, shape=beta_gamma_shape),
                                 name='beta', trainable=False)
    gamma = tf.Variable(tf.constant(1.0, shape=beta_gamma_shape),
                                  name='gamma', trainable=False)
    mean, var = tf.nn.moments(variable, axes, name='moments')

    ema_apply_op = batch_norm_ema.apply([mean, var])

    effective_mean, effective_var = tf.cond(is_training,
                                lambda: (mean, var),
                                lambda: (batch_norm_ema.average(mean), batch_norm_ema.average(var)))

    normed_variable = tf.nn.batch_normalization(variable, effective_mean, effective_var, beta, gamma, 1e-8)

    return normed_variable, ema_apply_op

# %%

with tf.device("/cpu:0"):
    word_embedding = tf.Variable(EMBEDDINGS, trainable=True, name='word_embedding', dtype=tf.float32)


set_word_embedding = tf_util.create_row_setter(word_embedding, "set_word_embedding")

# %%

with open('whereis_filters.pkl', 'rb') as f:
    filter_params = pickle.load(f)

# %%

NUM_BLOCKS = 256 # dg.max_blocks

with tf.variable_scope("block_embedding"):
    block_embedding = tf.Variable(filter_params['block_embedding'], trainable=False,
        name='block_embedding', dtype=tf.float32)

set_block_embedding = tf_util.create_row_setter(block_embedding, "set_block_embedding")

# %%

with tf.variable_scope("conv1_1"):
    filter_bank = tf.Variable(filter_params['filter_bank'],
        trainable=False,
        name="filter_bank",
        dtype=tf.float32)

voxels_padded = tf.pad(voxels % NUM_BLOCKS, [[0,0], [2,2], [2,2], [2,2]])

voxel_emb_1 = tf.nn.conv3d(tf.nn.embedding_lookup(block_embedding, voxels_padded ),
    filter_bank, strides=[1,1,1,1,1], padding="VALID")

voxel_emb_1s = tf.nn.sigmoid(voxel_emb_1)

# %%

with tf.variable_scope("conv2_1"):
    filter_bank_2 = tf.Variable(filter_params['filter_bank_2'],
        trainable=False,
        name="filter_bank_2",
        dtype=tf.float32)

voxel_emb_2 = tf.nn.conv3d(voxel_emb_1, filter_bank_2, strides=[1,1,1,1,1], padding="VALID")

normed_voxel_emb_2, voxel_op = batch_norm(voxel_emb_2, [0,1,2,3])
step_ops['voxel_op'] = voxel_op

voxel_emb_2s = tf.nn.sigmoid(normed_voxel_emb_2)

# %%

with tf.variable_scope('block_embedding_single'):
    block_embedding_single = tf.get_variable("block_embedding_single", [NUM_BLOCKS, word_emb_size],
        initializer=tf.zeros_initializer,
        trainable=True)

set_block_embedding_single = tf_util.create_row_setter(block_embedding_single, "set_block_embedding_single")

# %%

with tf.variable_scope('context_mapping'):
    context_mapping = tf.get_variable("context_mapping", [int(voxel_emb_2s.get_shape()[-1]), word_emb_size],
        initializer=tf.zeros_initializer,
        trainable=True)

voxel_vals_context_flat = tf.reshape(
    tf.matmul(tf.reshape(voxel_emb_2s, [-1, int(voxel_emb_2s.get_shape()[-1])]), context_mapping),
    [-1, VOXEL_VALS_SIZE**3, word_emb_size]
)

# %%

dynamic_batch_size = tf.shape(voxels)[0]

# %%

voxel_vals_single = tf.nn.embedding_lookup(block_embedding_single, voxels % NUM_BLOCKS)

# %%

ones_like_words = tf.SparseTensor(words.indices, tf.ones_like(words.values), words.shape)
words_len = tf.stop_gradient(tf.sparse_reduce_sum(ones_like_words, 1))
words_len.set_shape([batch_size])

words_dense = tf.sparse_tensor_to_dense(words)
words_dense.set_shape([batch_size, None])

words_dense

dense_length = tf.shape(words_dense)[1]

ones_like_words_dense = tf.sparse_tensor_to_dense(ones_like_words)
ones_like_words_dense.set_shape(words_dense.get_shape())

valid_timesteps_mask = tf.reshape(ones_like_words_dense, [dynamic_batch_size, -1, 1, 1, 1])

# %% (dropout/batchnorm in this section unused)

with tf.variable_scope('unk_word_val_2'):
    unk_word_val = tf.get_variable('unk_word_val',
        shape=[1, 1, word_embedding.get_shape()[1]],
        initializer=tf.zeros_initializer)

keep_word_vals = tf.expand_dims(tf.nn.dropout(tf.to_float(ones_like_words_dense),
    tf.select(is_training, 0.7, 1.0)) > 0.0, -1)
keep_word_vals = tf.tile(keep_word_vals, [1, 1, int(word_embedding.get_shape()[1])])

word_vals = tf.nn.embedding_lookup(word_embedding, words_dense)

with tf.variable_scope('word_batch_norm_2'):
    normed_word_vals, word_vals_op = batch_norm(word_vals, [0,1])

normed_word_vals = tf.select(keep_word_vals,
    normed_word_vals,
    tf.tile(unk_word_val, [dynamic_batch_size, dense_length, 1]))

step_ops['word_vals_op'] = word_vals_op

normed_word_vals

# %%

voxel_vals_context_flat_ = 0.1 * tf.select(
    global_step > 2321 * 2,
    voxel_vals_context_flat,
    tf.stop_gradient(voxel_vals_context_flat))

voxel_vals_context_flat

if ABLATION == 'single_block':
    voxel_vals_target_flat = tf.reshape(voxel_vals_single, [-1, VOXEL_VALS_SIZE**3, word_emb_size])
else:
    voxel_vals_target_flat = tf.reshape(voxel_vals_single, [-1, VOXEL_VALS_SIZE**3, word_emb_size]) + voxel_vals_context_flat_

# %%

match_logits = tf.batch_matmul(
    word_vals,
    voxel_vals_target_flat,
    adj_y=True
)

air_mask = tf.expand_dims(tf.reshape(tf.equal(voxels, 0), [-1, VOXEL_VALS_SIZE**3]), 1)

air_logits = tf.boolean_mask(
    tf.nn.log_softmax(match_logits),
    (tf.squeeze(valid_timesteps_mask, [-1,-2]) > 0) & air_mask)


air_logits_extra = tf.nn.relu(air_logits - np.log(1./VOXEL_VALS_SIZE**3))
air_logits_loss = tf.reduce_sum(air_logits_extra) / (VOXEL_VALS_SIZE**3) / tf.to_float(dynamic_batch_size)

match_logits_valid = tf.boolean_mask(match_logits,
    (~air_mask) & tf.reshape(valid_timesteps_mask > 0, [dynamic_batch_size, -1, 1]))

match_probs = tf.nn.softmax(match_logits)

match_probs_gap = tf.reduce_max(match_probs[:,-1,:]) - tf.reduce_min(match_probs[:,-1,:])

# %% calculate the values that will be used to compute flag and action probabilities

droppedout_word_vals = tf.nn.dropout(word_vals,
    tf.select(is_training, 0.5, 1.0),
    [tf.shape(word_vals)[0], tf.shape(word_vals)[1], 1])

if ABLATION != 'no_lstm':
    # LSTM features running over the entire sentence
    num_units_bw = 256
    num_units_fw = 256

    bw_cell = tf.nn.rnn_cell.LSTMCell(
        num_units=num_units_bw,
        use_peepholes=True)

    fw_cell = tf.nn.rnn_cell.LSTMCell(
        num_units=num_units_fw,
        use_peepholes=True)

    with tf.variable_scope('preaction_lstm_2') as scope:
        lstm_states, _ = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell,
            droppedout_word_vals,
            sequence_length=words_len,
            dtype=tf.float32,
            scope=scope
            )

    lstm_vals = tf.concat(2, lstm_states)

    preaction_vals = lstm_vals
else:
    masked_word_vals = droppedout_word_vals * tf.expand_dims(tf.to_float(ones_like_words_dense), -1)

    padded_word_vals = tf.pad(masked_word_vals, [[0,0],[1,1],[0,0]])

    word_context_vals = tf.concat(2, [
        padded_word_vals[:,2:,:],
        padded_word_vals[:,1:-1,:],
        padded_word_vals[:,:-2,:],
    ])

    with tf.variable_scope("word_context") as scope:
        preaction_vals = tf_util.linear(word_context_vals, 512, bias=True, scope=scope)

    preaction_vals = tf.nn.relu(preaction_vals)

preaction_vals

# %% utilities for creating radial and angular filters

def make_radial_filter(kernel_shape, radial_inputs):
    assert kernel_shape[0] % 2 == 1 and kernel_shape[1] % 2 == 1
    assert int(radial_inputs.get_shape()[1]) >= np.sqrt((kernel_shape[0] // 2)**2 + (kernel_shape[1] // 2)**2)
    assert radial_inputs.dtype not in [tf.int32, tf.int64]

    distances = np.zeros(kernel_shape)
    cx, cy = distances.shape[0] // 2, distances.shape[1] // 2
    for x in range(0, distances.shape[0]):
        for y in range(0, distances.shape[1]):
            distances[x, y] = (x - cx) ** 2 + (y - cy) ** 2

    distances = np.sqrt(distances)

    distances_floor = tf.constant(np.floor(distances), dtype=tf.int32)
    distances_rem = tf.expand_dims(tf.constant(distances % 1, dtype=tf.float32),0)

    radial_inputs_transpose = tf.transpose(radial_inputs)
    radial_filter_floor = tf.transpose(tf.gather(radial_inputs_transpose, distances_floor), (2,0,1))
    radial_filter_ceil = tf.transpose(tf.gather(radial_inputs_transpose, distances_floor + 1), (2,0,1))

    radial_filter = radial_filter_floor * (1. - distances_rem) + radial_filter_ceil * distances_rem
    return radial_filter

# make_radial_filter((5,5), tf.constant([[0,1,2,3,4.], [-1,-2,-3,-4,-5.]]))

def make_angular_filter(kernel_shape, angular_inputs):
    # TODO(nikita): don't assign a value to the landmark itself
    assert kernel_shape[0] % 2 == 1 and kernel_shape[1] % 2 == 1
    assert angular_inputs.dtype not in [tf.int32, tf.int64]

    angles = np.zeros(kernel_shape)
    cx, cy = angles.shape[0] // 2, angles.shape[1] // 2
    for x in range(0, angles.shape[0]):
        for y in range(0, angles.shape[1]):
            angles[x, y] = math.atan2(y-cy, x-cx)
            if angles[x,y] < 0:
                angles[x, y] = 2 * math.pi + angles[x, y]

    angle_idxs = angles / (2* math.pi) * int(angular_inputs.get_shape()[1])

    # Have the center point to a dummy location, which will be assigned zero
    # probability mass (highly negative score)
    angle_idxs[cx, cy] = int(angular_inputs.get_shape()[1] + 1)

    idxs_floor = tf.constant(np.floor(angle_idxs), dtype=tf.int32)
    idxs_rem = tf.expand_dims(tf.constant(angle_idxs % 1, dtype=tf.float32), 0)

    angular_inputs_wrapped = tf.transpose(tf.concat(1,
        [angular_inputs, angular_inputs[:,:1], tf.zeros_like(angular_inputs[:,:2])]))

    angular_filter_floor = tf.transpose(tf.gather(angular_inputs_wrapped, idxs_floor), (2,0,1))
    angular_filter_ceil = tf.transpose(tf.gather(angular_inputs_wrapped, idxs_floor + 1), (2,0,1))

    angular_filter = angular_filter_floor * (1. - idxs_rem) + angular_filter_ceil * idxs_rem

    return angular_filter

# make_angular_filter((5,5), tf.constant([[0,1,2,3.],[-1,-2,-3,-4.]])).eval()

#%% actual radial and angular filters

FILTER_SIZE = 3
ANGULAR_FILTER_SUBDIVISIONS = 4

# The number of elements is the radial filter need to be enough to cover all
# valid distances within the filter size
radial_filter_max_distance = math.ceil(math.sqrt(2) * (FILTER_SIZE // 2))
radial_filter_max_distance

if ABLATION == 'constant_filter':
    with tf.variable_scope("radial_probs_2", initializer=tf.zeros_initializer) as scope:
        radial_logits_flat = tf.get_variable("radial_logits", [1, radial_filter_max_distance + 1],
                trainable=True)

    with tf.variable_scope("angular_probs_2", initializer=tf.zeros_initializer) as scope:
        angular_logits_flat = tf.get_variable("radial_logits", [1, ANGULAR_FILTER_SUBDIVISIONS],
                trainable=True)

    with tf.variable_scope("height_probs_2", initializer=tf.zeros_initializer) as scope:
        height_logits = tf.get_variable("height_logits", [FILTER_SIZE],
                trainable=True)
else:
    with tf.variable_scope("radial_probs_2", initializer=tf.zeros_initializer) as scope:
        radial_logits = tf_util.linear(preaction_vals, radial_filter_max_distance + 1, bias=True, scope=scope)

    radial_logits_flat = tf.reshape(radial_logits, [-1, radial_filter_max_distance+1])

    with tf.variable_scope("angular_probs_2", initializer=tf.zeros_initializer) as scope:
        angular_logits = tf_util.linear(preaction_vals, ANGULAR_FILTER_SUBDIVISIONS, bias=True, scope=scope)

    angular_logits_flat = tf.reshape(angular_logits, [-1, ANGULAR_FILTER_SUBDIVISIONS])

    with tf.variable_scope("height_probs_2", initializer=tf.zeros_initializer) as scope:
        height_logits = tf_util.linear(preaction_vals, FILTER_SIZE, bias=True, scope=scope)

# none_logits are for landmark detection, so they always use context
with tf.variable_scope("none_probs_2", initializer=tf.zeros_initializer) as scope:
    none_logits = tf_util.linear(preaction_vals, 1, bias=True, scope=scope)

filter_radial_logits_flat = tf.expand_dims(make_radial_filter((FILTER_SIZE,FILTER_SIZE), radial_logits_flat), 2)
filter_angular_logits_flat = tf.expand_dims(make_angular_filter((FILTER_SIZE,FILTER_SIZE), angular_logits_flat), 2)
filter_height_logits_flat = tf.reshape(height_logits, [-1, 1, FILTER_SIZE, 1])
filter_none_logits_flat = tf.reshape(none_logits, [-1, 1, 1, 1])

# %%

center_mask = np.zeros((FILTER_SIZE, FILTER_SIZE, FILTER_SIZE), dtype=bool)
center_mask[(FILTER_SIZE//2,)*3] = True
center_mask = tf.constant(center_mask.reshape((1, FILTER_SIZE, FILTER_SIZE, FILTER_SIZE)), dtype=tf.float32)
noncenter_mask = 1. - center_mask

# %%

filter_logits_flat = (filter_angular_logits_flat
    + filter_radial_logits_flat + filter_height_logits_flat) * noncenter_mask + (
    -1e18 * center_mask
    )

filter_logits_flat = (filter_angular_logits_flat
    + filter_radial_logits_flat + filter_height_logits_flat) * noncenter_mask + (
    filter_none_logits_flat * center_mask
    )

filter_probs_flat_ = tf.reshape(tf.nn.softmax(tf.reshape(filter_logits_flat, [-1, FILTER_SIZE**3])),
    tf.shape(filter_logits_flat))

filter_probs_flat = filter_probs_flat_ * noncenter_mask

none_prob = tf.reshape(filter_probs_flat_[:,FILTER_SIZE//2,FILTER_SIZE//2, FILTER_SIZE//2], [dynamic_batch_size, dense_length])
none_prob.set_shape((batch_size, None))

# for debug only
filter_probs = tf.reshape(filter_probs_flat, [dynamic_batch_size, dense_length, FILTER_SIZE, FILTER_SIZE, FILTER_SIZE])

# %%

def apply_conv(elems_bt):
    inp_bt, filter_bt = elems_bt
    return tf.nn.conv3d(
        inp_bt,
        filter_bt,
        strides=[1,1,1,1,1],
        padding="SAME")

filtered_mask_logits = tf.squeeze(tf.map_fn(apply_conv, [
    tf.reshape(match_logits, [-1, 1, VOXEL_VALS_SIZE, VOXEL_VALS_SIZE, VOXEL_VALS_SIZE, 1]),
    tf.reshape(filter_probs_flat, [-1, FILTER_SIZE, FILTER_SIZE, FILTER_SIZE, 1, 1])],
    dtype=tf.float32), [1, 5])

filtered_mask_logits = tf.reshape(filtered_mask_logits, [dynamic_batch_size, dense_length, VOXEL_VALS_SIZE, VOXEL_VALS_SIZE, VOXEL_VALS_SIZE])

filtered_mask = tf.reshape(
    tf.nn.softmax(tf.reshape(filtered_mask_logits, [dynamic_batch_size, dense_length, VOXEL_VALS_SIZE**3])),
    [dynamic_batch_size, dense_length, VOXEL_VALS_SIZE, VOXEL_VALS_SIZE, VOXEL_VALS_SIZE])

# %% generate a mask for each timestep in accordance with the actions chosen

flat_none_probs = tf.boolean_mask(none_prob, ones_like_words_dense > 0)
none_mean, none_variance = tf.nn.moments(
    flat_none_probs,
    [0])
none_std = tf.sqrt(none_variance)
none_max = tf.reduce_max(flat_none_probs)
none_min = tf.reduce_min(flat_none_probs)

per_timestep_mask_logits = filtered_mask_logits * tf.stop_gradient(tf.to_float(valid_timesteps_mask))

per_timestep_mask_valid = tf.boolean_mask(per_timestep_mask_logits,
    tf.expand_dims(candidates_mask, 1) & tf.reshape(valid_timesteps_mask > 0, [dynamic_batch_size, -1, 1, 1, 1]))

per_timestep_mask_flat = tf.reshape(per_timestep_mask_logits, [dynamic_batch_size, dense_length, VOXEL_VALS_SIZE**3])

per_timestep_entdiff = -np.log(1./19**3) + tf.reduce_sum(tf.nn.log_softmax(per_timestep_mask_flat) * tf.nn.softmax(per_timestep_mask_flat), -1)
per_timestep_entdiff = per_timestep_entdiff * tf.to_float(tf.squeeze(valid_timesteps_mask, [-1,-2,-3]))

per_timestep_mask_cands_flat = tf.reshape(
    per_timestep_mask_logits - 1e5 * (1. - tf.to_float(tf.expand_dims(candidates_mask, 1))),
    [dynamic_batch_size, dense_length, VOXEL_VALS_SIZE**3])

per_timestep_cands_entdiff = -tf.expand_dims(tf.log(1. / tf.reduce_sum(tf.to_float(candidates_mask), [-1,-2,-3])),
    -1) + tf.reduce_sum(tf.nn.log_softmax(per_timestep_mask_cands_flat
    ) * tf.nn.softmax(per_timestep_mask_cands_flat), -1)
per_timestep_cands_entdiff = per_timestep_cands_entdiff * tf.to_float(tf.squeeze(valid_timesteps_mask, [-1,-2,-3]))


# per_timestep_mask.eval().shape
# %%

OUTPUT_SIDE = VOXEL_VALS_SIZE

# TODO(nikita): resolve how to include the candidates mask
output_block_logits_denorm = tf.reduce_sum(per_timestep_mask_logits, reduction_indices=1)
output_block_logits_flat = tf.reshape(output_block_logits_denorm, [-1, OUTPUT_SIDE**3])
output_block_logits_clipped_flat = output_block_logits_flat - 1e2 * (1. - tf.to_float(tf.reshape(candidates_mask, [-1, OUTPUT_SIDE**3])))

# for debug only
output_block_probs = tf.reshape(tf.nn.softmax(output_block_logits_flat), [-1, OUTPUT_SIDE, OUTPUT_SIDE, OUTPUT_SIDE])
output_block_probs_clipped = output_block_probs * tf.to_float(candidates_mask)
output_block_probs_clipped_2 = tf.reshape(tf.nn.softmax(output_block_logits_clipped_flat), [-1, OUTPUT_SIDE, OUTPUT_SIDE, OUTPUT_SIDE])


misty_idxs = ((misty_location[:,0] - (19 - OUTPUT_SIDE)//2) * OUTPUT_SIDE * OUTPUT_SIDE
            + (misty_location[:,1] - (19 - OUTPUT_SIDE)//2) * OUTPUT_SIDE
            + (misty_location[:,2] - (19 - OUTPUT_SIDE)//2))

# example_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(output_block_logits_clipped_flat, misty_idxs)
example_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(output_block_logits_flat, misty_idxs)

loss = tf.reduce_mean(example_losses)
loss += air_logits_loss

example_errors = tf.not_equal(
    tf.argmax(output_block_logits_clipped_flat, 1),
    misty_idxs
)

error = tf.reduce_mean(tf.to_float(example_errors))

# %%
loss_summary = tf.scalar_summary("train/loss", loss)
error_summary = tf.scalar_summary("train/error", error)

valid_loss_summary = tf.scalar_summary("valid/loss", loss, collections=['valid_summaries'])
valid_error_summary = tf.scalar_summary("valid/error", error, collections=['valid_summaries'])
# %%
lr = tf.Variable(0.001, dtype=tf.float32)
lr_summary = tf.scalar_summary("train/lr", lr)

set_lr = tf_util.create_var_setter(lr)
# %%

opt = tf.train.AdamOptimizer(lr)

grads_and_vars = opt.compute_gradients(loss)

apply_gradients_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
step_ops['apply_gradients_op'] = apply_gradients_op

# %%

ema = tf.train.ExponentialMovingAverage(decay=0.999)
step_ops['maintain_averages_op'] = ema.apply(tf.trainable_variables())

saver_ema = tf.train.Saver(ema.variables_to_restore())

# %%

with tf.control_dependencies(list(step_ops.values())):
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
        words: words,
        candidates_mask: candidates_mask,
        misty_location: misty_location,
        example_id: example_id,
    })
    feed[is_training] = False
    train_feeds.append(feed)

sess.run(loss, train_feeds[0])

# %%

def eval_model(feeds=None):
    if feeds is None:
        feeds = train_feeds
    elif not feeds:
        return 0.0, 0.0
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

# eval_model()

# %%

def get_words(feed_dict={is_training: False}, idx=0):
    vals = tf_util.safe_run(sess, {
        'example_id': example_id,
        'words':words,
        }, feed_dict)
    vals['words'] = tf_util.sparse_to_list(vals['words'])
    # Limit ourselves to the first example in the batch
    for k in vals:
        vals[k] = vals[k][idx]

    return " ".join([VOCAB[i] for i in vals['words']])

def inspect_one(feed_dict={is_training: False}, idx=0, full_range=False, task=None):
    global _last_task

    # This import is not needed for non-interactive mode
    from task_tools import whereis_inspection
    if task is not None:
        whereis_inspection.activate_task(task)
        return

    vals = tf_util.safe_run(sess, {
        'example_id': example_id,
        'voxels':voxels,
        'words':words,
        'misty_location': misty_location,
        'candidates_mask': candidates_mask,
        'match_probs':match_probs,
        'filter_probs': filter_probs,
        'filtered_mask': filtered_mask,
        'none_prob': none_prob,
        'output_block_probs':output_block_probs,
        'output_block_probs_clipped':output_block_probs_clipped,
        'error': example_errors,
        'per_timestep_entdiff': per_timestep_entdiff,
        'per_timestep_cands_entdiff': per_timestep_cands_entdiff,
        }, feed_dict)
    vals['words'] = tf_util.sparse_to_list(vals['words'])
    # Limit ourselves to the first example in the batch
    for k in vals:
        vals[k] = vals[k][idx]

    vals['words_str'] = [VOCAB[i] for i in vals['words']]

    offset = vals['misty_location']

    print('Error is', vals['error'])
    print('Sentence is', vals['words_str'])

    if isinstance(dg, DataReader):
        task = whereis_inspection.get_task_from_example_id(vals['example_id']).result()
    else:
        task = whereis_inspection.get_task_from_values(
            *[vals[k] for k in "voxels words_str candidates_mask misty_location example_id".split()])

    # None probs
    words_len_val = len(vals['words'])
    print('None probs:', vals['none_prob'][:words_len_val])

    print("Joint:", " ".join("{}/{:1.3f}".format(w, p) for w,p in zip(vals['words_str'], 1. - vals['none_prob'][:words_len_val])))
    print("Entropy diff:", " ".join("{}/{:2.3f}".format(w, p) for w,p in zip(vals['words_str'], 100. * vals['per_timestep_entdiff'][:words_len_val])))
    print("Cands ent. diff:", " ".join("{}/{:2.3f}".format(w, p) for w,p in zip(vals['words_str'], 100. * vals['per_timestep_cands_entdiff'][:words_len_val])))

    whereis_inspection.add_misty_relative_heatmap(task,
        'candidates_mask',
        vals['candidates_mask'],
        offset
        )

    # Per-word match probability
    for i, word_idx in enumerate(vals['words']):
        word = VOCAB[word_idx]
        action_prob = 1. - vals['none_prob'][i]
        # if action_prob < 0.3:
            # continue
        match_probs_val = vals['match_probs'][i].reshape((VOXEL_VALS_SIZE,)*3)
        if not full_range:
            match_probs_val /= np.max(match_probs_val) or 1.
        whereis_inspection.add_misty_relative_heatmap(task,
            'match_probs/{:02}/{}'.format(i, word),
            match_probs_val,
            offset
            )

    # Display "key" filters that are in-use
    for i, word_idx in enumerate(vals['words']):
        word = VOCAB[word_idx]
        action_prob = 1. - vals['none_prob'][i]
        if True: #action_prob > 0.3:
            filter_vals = vals['filter_probs'][i]
            filter_vals /= np.sum(filter_vals) or 1.
            whereis_inspection.add_misty_relative_heatmap(task,
                'filter/{:02}/{}'.format(i, word),
                filter_vals[::-1,::-1,::-1]
                )

    for i, word_idx in enumerate(vals['words']):
        word = VOCAB[word_idx]
        action_prob = 1. - vals['none_prob'][i]
        if True: #action_prob > 0.3:
            mask_clipped = vals['filtered_mask'][i] * vals['candidates_mask']
            mask_clipped /= np.sum(mask_clipped) or 1.

            whereis_inspection.add_misty_relative_heatmap(task,
                'mask/{:02}/{}_clipped'.format(i, word),
                mask_clipped,
                offset
                )

    for i, word_idx in enumerate(vals['words']):
        word = VOCAB[word_idx]
        action_prob = 1. - vals['none_prob'][i]
        if True: #action_prob > 0.3:
            mask_clipped = vals['filtered_mask'][i]
            mask_clipped /= np.sum(mask_clipped) or 1.

            whereis_inspection.add_misty_relative_heatmap(task,
                'mask/{:02}/{}'.format(i, word),
                mask_clipped,
                offset
                )

    output_block_probs_val = vals['output_block_probs'].reshape((OUTPUT_SIDE,)*3)
    if not full_range:
        output_block_probs_val /= np.max(output_block_probs_val) or 1.

    whereis_inspection.add_misty_relative_heatmap(task,
        'output_block_probs',
        output_block_probs_val,
        offset
        )

    output_block_probs_clipped_val = vals['output_block_probs_clipped'].reshape((OUTPUT_SIDE,)*3)
    if not full_range:
        output_block_probs_clipped_val /= np.max(output_block_probs_clipped_val) or 1.

    whereis_inspection.add_misty_relative_heatmap(task,
        'output_block_probs_clipped',
        output_block_probs_clipped_val,
        offset
        )

    whereis_inspection.activate_task(task, from_values=(not isinstance(dg, DataReader)))

    _last_task = task


# %%

def plot_spheres(feed_dict={is_training: False}, idx=0, wnum=0):
    vals = tf_util.safe_run(sess, {
        'example_id': example_id,
        'voxels':voxels,
        'words':words,
        'misty_location': misty_location,
        'candidates_mask': candidates_mask,
        'match_probs':match_probs,
        'filter_probs': filter_probs,
        'filtered_mask': filtered_mask,
        'none_prob': none_prob,
        'output_block_probs':output_block_probs,
        'output_block_probs_clipped':output_block_probs_clipped,
        'error': example_errors
        }, feed_dict)
    vals['words'] = tf_util.sparse_to_list(vals['words'])
    # Limit ourselves to the first example in the batch
    for k in vals:
        vals[k] = vals[k][idx]

    fp = vals['filter_probs'][wnum]

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm

    fig = plt.figure(figsize=(12,12), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)


    def draw(ax, ox, oy, oz, rad, color='b', alpha=1.):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = ox + rad * np.outer(np.cos(u), np.sin(v))
        y = oy + rad * np.outer(np.sin(u), np.sin(v))
        z = oz + rad * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color, linewidth=0, alpha=alpha)

    fp /= np.max(fp)

    colors = matplotlib.cm.get_cmap('Blues')

    for xx in range(3):
        for yy in range(3):
            for zz in range(3):
                if xx==1 and yy==1 and zz==1:
                    # in minecraft coords, +y is up
                    draw(ax, 1-xx, zz-1, 1-yy, 0.2, color='r')
                    # ax.scatter([1-xx], [zz-1], [1-yy], c='r', color='r', marker='D', s=250)
                    continue

                alpha = fp[xx][yy][zz]
                # if alpha > 0.02:
                if True:
                    # in minecraft coords, +y is up
                    draw(ax, 1-xx, zz-1, 1-yy, 0.3, color=colors(fp[xx][yy][zz]))#color=(0.,0.,fp[xx][yy][zz],1.))#, alpha=fp[xx][yy][zz])


# %%

if EVALUATE:
    saver_ema.restore(sess, 'model/model-16240')

if EVALUATE and False:
    # Skip this for release code, for now
    best_cp_error, best_cp_path = 1.0, None
    min_delta = 0.8 / (len(dev_feeds) * batch_size)
    for checkpoint_path in sorted(glob.glob(SAVER_DIR + '/model-*.index'),
            key=lambda x: int(x.split('-')[-1].split('.')[0])):
        checkpoint_path = checkpoint_path.rstrip('.index')
        saver_ema.restore(sess, checkpoint_path)
        this_error, this_loss = eval_model(dev_feeds)
        print(checkpoint_path.split('/')[-1], this_error, this_loss)
        if this_error + min_delta < best_cp_error:
            best_cp_error = this_error
            best_cp_path = checkpoint_path
    saver_ema.restore(sess, best_cp_path)

# %%

if EVALUATE:
    print('Restored saved model!')
    print("Dev set error/loss:", eval_model(dev_feeds))
    print("Test set error/loss:", eval_model(test_feeds))

# %%

if EVALUATE and not INTERACTIVE:
    sys.exit(0)

# %%
assert not EVALUATE

varclear()

log = {}
num_too_big = 0

if global_step.eval() in [0, 1]:
    saver = tf.train.Saver(max_to_keep=100)

    if INTERACTIVE:
        get_ipython().system('mkdir -p $SAVER_DIR')
        get_ipython().system('ls $SAVER_DIR')
    print('saving...')
    saver.save(sess, SAVER_DIR + '/model', global_step=global_step.eval())

# %%

print("Starting training")
print("SAVER_DIR =", SAVER_DIR)
print("ABLATION =", ABLATION)
print()

i = global_step.eval()

while i < 23210: # 100 epochs
    try:
        if i % 25 == 0:
            log[i] = (eval_model(), eval_model(dev_feeds),
                sess.run(match_probs_gap, train_feeds[0]),
                sess.run((none_mean, none_std, none_min, none_max), train_feeds[0]))
            print(i, *log[i])
        if i % 464 == 0:
            saver.save(sess, SAVER_DIR + '/model', global_step=i)

        sess.run(train_op)
        i += 1
    except tf.errors.ResourceExhaustedError:
        print('ERROR: could not fit into memory')
        num_too_big += 1
        continue

# %%

if INTERACTIVE:
    for k in sorted(log.keys()):
        print(k, *log[k])

# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
