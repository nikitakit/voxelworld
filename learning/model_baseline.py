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

SAVER_DIR = os.path.expanduser('~/tf_logs/{}'.format('whereis_f_baseline_2'))
EVALUATE = True

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
# dg = RandomRoomData()
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

NUM_BLOCKS = 256 # dg.max_blocks
block_emb_size = 256
layer1_size = 256
layer2_size = 256

with tf.variable_scope("block_embedding"):
    block_embedding = tf.get_variable("block_embedding_b", [NUM_BLOCKS, block_emb_size],
        trainable=True)

set_block_embedding = tf_util.create_row_setter(block_embedding, "set_block_embedding")

# %%

with tf.variable_scope("conv1_1"):
    filter_bank = tf.get_variable("filter_bank", [2, 2, 2, block_emb_size, layer1_size], trainable=True)

voxels_padded = tf.pad(voxels % NUM_BLOCKS, [[0,0], [2,2], [2,2], [2,2]])

voxel_emb_1 = tf.nn.conv3d(tf.nn.embedding_lookup(block_embedding, voxels_padded ),
    filter_bank, strides=[1,1,1,1,1], padding="VALID")

voxel_emb_1s = tf.nn.sigmoid(voxel_emb_1)

# %%

with tf.variable_scope("conv2_1"):
    filter_bank_2 = tf.get_variable("filter_bank", [4, 4, 4, layer1_size, layer2_size], trainable=True)

voxel_emb_2 = tf.nn.conv3d(voxel_emb_1, filter_bank_2, strides=[1,1,1,1,1], padding="VALID")

normed_voxel_emb_2, voxel_op = batch_norm(voxel_emb_2, [0,1,2,3])
step_ops['voxel_op'] = voxel_op

voxel_emb_2s = tf.nn.sigmoid(normed_voxel_emb_2)

voxel_vals_flat =  tf.reshape(voxel_emb_2s, [-1, VOXEL_VALS_SIZE**3, layer2_size])

# %%

dynamic_batch_size = tf.shape(voxels)[0]

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

# %%

air_mask = tf.expand_dims(tf.reshape(tf.equal(voxels, 0), [-1, VOXEL_VALS_SIZE**3]), 1)

# %% calculate the values that will be used to compute flag and action probabilities

word_vals = tf.nn.embedding_lookup(word_embedding, words_dense)

droppedout_word_vals = tf.nn.dropout(word_vals,
    tf.select(is_training, 0.5, 1.0),
    [tf.shape(word_vals)[0], tf.shape(word_vals)[1], 1])

# LSTM features running over the entire sentence
num_units_bw = num_units_fw = (layer2_size // 2)

bw_cell = tf.nn.rnn_cell.LSTMCell(
    num_units=num_units_bw)

fw_cell = tf.nn.rnn_cell.LSTMCell(
    num_units=num_units_fw)

with tf.variable_scope('preaction_lstm_3') as scope:
    lstm_output, lstm_output_states = tf.nn.bidirectional_dynamic_rnn(
        fw_cell, bw_cell,
        droppedout_word_vals,
        sequence_length=words_len,
        dtype=tf.float32,
        scope=scope
        )

lstm_vals = tf.concat(1, [lstm_output_states[0].c, lstm_output_states[1].c])

lstm_vals

# %%

output_block_logits_flat = tf.batch_matmul(
    tf.expand_dims(lstm_vals, 1),
    voxel_vals_flat,
    adj_y=True
)

output_block_logits_flat = tf.squeeze(output_block_logits_flat, 1)

# %%

OUTPUT_SIDE = VOXEL_VALS_SIZE

# TODO(nikita): resolve how to include the candidates mask
output_block_logits_clipped_flat = output_block_logits_flat - 1e2 * (1. - tf.to_float(tf.reshape(candidates_mask, [-1, OUTPUT_SIDE**3])))

# for debug only
output_block_probs = tf.reshape(tf.nn.softmax(output_block_logits_flat), [-1, OUTPUT_SIDE, OUTPUT_SIDE, OUTPUT_SIDE])
output_block_probs_clipped = tf.reshape(tf.nn.softmax(output_block_logits_clipped_flat), [-1, OUTPUT_SIDE, OUTPUT_SIDE, OUTPUT_SIDE])


misty_idxs = ((misty_location[:,0] - (19 - OUTPUT_SIDE)//2) * OUTPUT_SIDE * OUTPUT_SIDE
            + (misty_location[:,1] - (19 - OUTPUT_SIDE)//2) * OUTPUT_SIDE
            + (misty_location[:,2] - (19 - OUTPUT_SIDE)//2))

# example_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(output_block_logits_clipped_flat, misty_idxs)
example_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(output_block_logits_flat, misty_idxs)

loss = tf.reduce_mean(example_losses)

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

eval_model()

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
        'output_block_probs':output_block_probs,
        'output_block_probs_clipped':output_block_probs_clipped,
        'error': example_errors,
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

    whereis_inspection.add_misty_relative_heatmap(task,
        'candidates_mask',
        vals['candidates_mask'],
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

# inspect_one()

# %%

if EVALUATE:
    best_cp_error, best_cp_path = 1.0, None
    min_delta = 0.8 / (len(dev_feeds) * batch_size)
    for checkpoint_path in sorted(glob.glob(SAVER_DIR + '/model-*.index'),
            key=lambda x: int(x.split('-')[-1].split('.')[0])):
        checkpoint_path = checkpoint_path.rstrip('.index')
        # saver.restore(sess, checkpoint_path)
        saver_ema.restore(sess, checkpoint_path)
        this_error, this_loss = eval_model(dev_feeds)
        print(checkpoint_path.split('/')[-1], this_error, this_loss)
        if this_error + min_delta < best_cp_error:
            best_cp_error = this_error
            best_cp_path = checkpoint_path
    saver_ema.restore(sess, best_cp_path)

# %%

if EVALUATE:
    print("Dev set error/loss:", eval_model(dev_feeds))

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
print("MODEL = baseline")
print()

i = global_step.eval()

while i < 23210: # 100 epochs
    try:
        if i % 25 == 0:
            log[i] = (eval_model(), eval_model(dev_feeds))
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
