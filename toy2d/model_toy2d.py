%cd ~/dev/mctest/toy2d
import xdbg

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

from data_toy2d import GrammarGeneratedData, DataReader
from data_toy2d import draw_voxels, draw_heatmap

from util import tf_util

# %%

from task_tools import whereis_inspection


from matplotlib import pyplot as plt
%matplotlib inline

# %%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# %%
global_step = tf.contrib.framework.get_or_create_global_step()

# %%
batch_size = 10

GRAMMAR="""
1.0 ROOT -> S | random_fill_some
# TODO(nikita): I think that due to rejection sampling this is not actually a 30-70 split
0.3 S -> CLAUSE
0.7 S -> COMPOUND_CLAUSE
0.25 CLAUSE -> ABOVE
0.25 CLAUSE -> BELOW
0.25 CLAUSE -> LEFT
0.25 CLAUSE -> RIGHT
1.0 ABOVE -> above the @block | above_far matching_distractors
1.0 BELOW -> below the @block | below_far matching_distractors
1.0 LEFT -> left of the @block | left_far matching_distractors
1.0 RIGHT -> right of the @block | right_far matching_distractors
0.1 COMPOUND_CLAUSE -> above the @block that is above the @block | above_above compound_distractors
0.075 COMPOUND_CLAUSE -> above the @block that is left of the @block | above_left compound_distractors
0.075 COMPOUND_CLAUSE -> above the @block that is right of the @block | above_right compound_distractors
0.1 COMPOUND_CLAUSE -> below the @block that is below the @block | below_below compound_distractors
0.075 COMPOUND_CLAUSE -> below the @block that is left of the @block | below_left compound_distractors
0.075 COMPOUND_CLAUSE -> below the @block that is right of the @block | below_right compound_distractors
0.075 COMPOUND_CLAUSE -> left of the @block that is above the @block | left_above compound_distractors
0.075 COMPOUND_CLAUSE -> left of the @block that is below of the @block | left_below compound_distractors
0.1 COMPOUND_CLAUSE -> left of the @block that is left of the @block | left_left compound_distractors
0.075 COMPOUND_CLAUSE -> right of the @block that is above the @block | right_above compound_distractors
0.075 COMPOUND_CLAUSE -> right of the @block that is below of the @block | right_below compound_distractors
0.1 COMPOUND_CLAUSE -> right of the @block that is right of the @block | right_right compound_distractors
"""

# dg = DataReader("responses.tfrecords")
dg = GrammarGeneratedData(grammar=GRAMMAR)
voxels, words, candidates_mask, misty_location, example_id = dg.get_inputs(batch_size=batch_size)

coord = tf.train.Coordinator()
coord._my_threads = (list(dg.create_threads(sess, coord, start=True))
                     + list(tf.train.start_queue_runners(sess=sess, coord=coord)))
coord.should_stop()

test_feeds = dg.get_test_feeds()

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
word_embedding = tf.Variable(EMBEDDINGS, trainable=True, name='word_embedding', dtype=tf.float32)

set_word_embedding = tf_util.create_row_setter(word_embedding, "set_word_embedding")
def preset_word_embeddings():
    word_emb_val = np.zeros((len(VOCAB), EMBEDDINGS.shape[1]))
    word_emb_val[:,:len(VOCAB)] = np.eye(len(VOCAB))

    set_word_embedding(
        list(range(len(VOCAB))),
        word_emb_val
        )

# %%

NUM_BLOCKS = dg.max_blocks
word_emb_size = EMBEDDINGS.shape[1]
position_emb_size = 2
block_emb_size = word_emb_size

sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
initializer = tf.zeros_initializer
#initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

with tf.device("/cpu:0"):
    block_embedding = tf.get_variable("block_embedding", [NUM_BLOCKS, block_emb_size], initializer=initializer,
                               trainable=True)

set_block_embedding = tf_util.create_row_setter(block_embedding, "set_block_embedding")

def preset_block_embeddings(norm=1.0):
    for i, row_value in enumerate(word_embedding.eval()):
        set_block_embedding([i], [row_value / np.linalg.norm(row_value) * norm])

# %%
dynamic_batch_size = tf.shape(voxels)[0]

# %%

voxel_embedded_vals = tf.nn.embedding_lookup(block_embedding, voxels)

# If no conv layer here
voxel_vals_target = voxel_embedded_vals
voxel_vals_target_flat = tf.reshape(voxel_vals_target, [-1, 19*19, word_emb_size])

voxel_vals_target

VOXEL_VALS_SIZE = 19

# %%

ones_like_words = tf.SparseTensor(words.indices, tf.ones_like(words.values), words.shape)
words_len = tf.stop_gradient(tf.sparse_reduce_sum(ones_like_words, 1))
words_len.set_shape([batch_size])

words_dense = tf.sparse_tensor_to_dense(words)
words_dense.set_shape([batch_size, None])

words_dense

word_vals = tf.nn.embedding_lookup(word_embedding, words_dense)

# %%
match_logits = tf.batch_matmul(
    word_vals,
    voxel_vals_target_flat,
    adj_y=True
)

air_mask = tf.expand_dims(tf.reshape(tf.equal(voxels, 0), [-1, VOXEL_VALS_SIZE**2]), 1)

match_logits = match_logits - 1e5 * tf.to_float(air_mask)

match_probs = tf.nn.softmax(match_logits)

match_probs_gap = tf.reduce_max(match_probs[:,2,:]) - tf.reduce_min(match_probs[:,2,:])
#match_probs_gap = tf.reduce_max(match_probs) - tf.reduce_min(match_probs)

#matching_blocks = tf.reduce_mean(match_probs, [1])

# %% calculate the values that will be used to compute flag and action probabilities

dense_length = tf.shape(words_dense)[1]

# Special features for word position, but only if it's one of the first 10 words
position_vals_uptoten = tf.tile(tf.expand_dims(tf.diag([1.,]*10), 0), [dynamic_batch_size,1,1])
position_vals = tf.concat(1, [position_vals_uptoten, tf.zeros((dynamic_batch_size, dense_length, 10))])[:,:dense_length,:]
position_vals.set_shape((None, None, 10))

# LSTM features running over the entire sentence
num_units_bw = 128
num_units_fw = 128

bw_cell = tf.nn.rnn_cell.LSTMCell(
    num_units=num_units_bw,
    use_peepholes=True)

fw_cell = tf.nn.rnn_cell.LSTMCell(
    num_units=num_units_fw,
    use_peepholes=True)

with tf.variable_scope('preaction_lstm_2') as scope:
    lstm_states, _ = tf.nn.bidirectional_dynamic_rnn(
        fw_cell, bw_cell,
        word_vals,
        sequence_length=words_len,
        dtype=tf.float32,
        scope=scope
        )

lstm_vals = tf.concat(2, lstm_states)

first_word_vals = tf.tile(word_vals[:,0:1,:], [1, dense_length, 1])

landmark_word_mask = tf.concat(1, [tf.zeros_like(first_word_vals[:,:-1,:]),
                                   tf.ones_like(first_word_vals[:,-1:,:])])

preaction_vals = lstm_vals
# preaction_vals = tf.concat(2, [first_word_vals, lstm_vals]) # works
# preaction_vals = tf.concat(2, [first_word_vals * landmark_word_mask, position_vals]) # works
# preaction_vals = tf.concat(2, [first_word_vals * landmark_word_mask, lstm_vals])
# preaction_vals = tf.concat(2, [first_word_vals, lstm_vals, position_vals])

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

FILTER_SIZE = 5
ANGULAR_FILTER_SUBDIVISIONS = 4

# The number of elements is the radial filter need to be enough to cover all
# valid distances within the filter size
radial_filter_max_distance = math.ceil(math.sqrt(2) * (FILTER_SIZE // 2))
radial_filter_max_distance

with tf.variable_scope("radial_probs_2", initializer=tf.zeros_initializer) as scope:
    radial_logits = tf_util.linear(preaction_vals, radial_filter_max_distance + 1, bias=False, scope=scope)

radial_logits_flat = tf.reshape(radial_logits, [-1, radial_filter_max_distance+1])

with tf.variable_scope("angular_probs_2", initializer=tf.zeros_initializer) as scope:
    angular_logits = tf_util.linear(preaction_vals, ANGULAR_FILTER_SUBDIVISIONS, bias=False, scope=scope)

angular_logits_flat = tf.reshape(angular_logits, [-1, ANGULAR_FILTER_SUBDIVISIONS])

filter_radial_logits_flat = make_radial_filter((FILTER_SIZE,FILTER_SIZE), radial_logits_flat)
filter_angular_logits_flat = make_angular_filter((FILTER_SIZE,FILTER_SIZE), angular_logits_flat)

# %%

center_mask = np.zeros((FILTER_SIZE, FILTER_SIZE), dtype=bool)
center_mask[(FILTER_SIZE//2,)*2] = True
center_mask = tf.constant(center_mask.reshape((1, FILTER_SIZE, FILTER_SIZE)), dtype=tf.float32)

# %%

filter_logits_flat = (filter_angular_logits_flat
    + filter_radial_logits_flat)

filter_probs_flat_ = tf.reshape(tf.nn.softmax(tf.reshape(filter_logits_flat, [-1, FILTER_SIZE**2])),
    tf.shape(filter_logits_flat))

filter_probs_flat = filter_probs_flat_ * (1. - center_mask)

filter_probs_flat

none_prob = tf.reshape(filter_probs_flat_[:,FILTER_SIZE//2,FILTER_SIZE//2], [dynamic_batch_size, dense_length])
none_prob.set_shape((batch_size, None))

# for debug only
filter_probs = tf.reshape(filter_probs_flat, [dynamic_batch_size, dense_length, FILTER_SIZE, FILTER_SIZE])

# %%

filtered_mask = tf.nn.depthwise_conv2d(
    tf.reshape(tf.transpose(match_probs, (2,0,1)), [1, VOXEL_VALS_SIZE, VOXEL_VALS_SIZE, -1]),
    tf.expand_dims(tf.transpose(filter_probs_flat, (1,2,0)), 3),
    strides=[1,1,1,1],
    padding="SAME")
filtered_mask = tf.reshape(tf.transpose(filtered_mask, (0,3,1,2)),
    [dynamic_batch_size, dense_length, VOXEL_VALS_SIZE, VOXEL_VALS_SIZE])

# %% generate a mask for each timestep in accordance with the actions chosen

none_mask_weighted = tf.expand_dims(tf.expand_dims(none_prob, 2), 3) * (1 / VOXEL_VALS_SIZE**2)

filtered_mask_weighted = filtered_mask

per_timestep_mask = none_mask_weighted + filtered_mask_weighted

valid_timesteps_mask = tf.reshape(tf.sparse_tensor_to_dense(ones_like_words), [dynamic_batch_size, dense_length, 1, 1])

per_timestep_mask_logits = tf.log(per_timestep_mask) * tf.stop_gradient(tf.to_float(valid_timesteps_mask))
per_timestep_mask_logits

# per_timestep_mask.eval().shape
# %%

OUTPUT_SIDE = VOXEL_VALS_SIZE

# TODO(nikita): resolve how to include the candidates mask
output_block_logits_denorm = tf.reduce_sum(per_timestep_mask_logits, reduction_indices=1)
output_block_logits_flat = tf.reshape(output_block_logits_denorm, [-1, OUTPUT_SIDE**2])
output_block_logits_flat -= 1e2 * (1. - tf.to_float(tf.reshape(candidates_mask, [-1, OUTPUT_SIDE**2])))

# for debug only
output_block_probs = tf.reshape(tf.nn.softmax(output_block_logits_flat), [-1, OUTPUT_SIDE, OUTPUT_SIDE])

misty_idxs = (misty_location[:,0] - (19 - OUTPUT_SIDE)//2) * OUTPUT_SIDE + (misty_location[:,1] - (19 - OUTPUT_SIDE)//2)

example_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(output_block_logits_flat, misty_idxs)
# example_losses = tf.select(tf.less_equal(example_losses, 1e20), example_losses, tf.zeros_like(example_losses))

loss = tf.reduce_mean(example_losses)

example_errors = tf.not_equal(
    tf.argmax(output_block_logits_flat, 1),
    misty_idxs
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

emb_lr = tf.Variable(0.5, dtype=tf.float32)
set_emb_lr = tf_util.create_var_setter(emb_lr)

opt_block_embeddings = tf.train.MomentumOptimizer(emb_lr, 0.95)
opt = tf.train.AdadeltaOptimizer(lr)

grads_and_vars = opt.compute_gradients(loss)

apply_gradients_op = opt.apply_gradients(
    [(g,v) for (g,v) in grads_and_vars if v != block_embedding],
    global_step=global_step)

apply_emb_gradients_op = opt_block_embeddings.apply_gradients(
    [(g,v) for (g,v) in grads_and_vars if v == block_embedding])

with tf.control_dependencies([apply_gradients_op,apply_emb_gradients_op]):
    train_op = tf.no_op(name="train")

# %%

# opt = tf.train.AdadeltaOptimizer(lr)
# apply_gradients_op = opt.minimize(loss, global_step=global_step)
#
# with tf.control_dependencies([apply_gradients_op,]):
#     train_op = tf.no_op(name="train")

# %%
merged_summary = tf.merge_all_summaries()

merged_valid_summary = tf.merge_all_summaries('valid_summaries')

# %%

varinit()

# %%

train_feeds = []
for count in range(50):
    feed = sess.run({
        voxels: voxels,
        words: words,
        candidates_mask: candidates_mask,
        misty_location: misty_location,
        example_id: example_id
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

NUM_BLOCKS_PLOT = dg.max_blocks

def draw_filter(title, vals):
    vals = np.transpose(vals)[:,::-1]
    voxelsize = 8. / VOXEL_VALS_SIZE
    plt.figure(figsize=(voxelsize * vals.shape[0], voxelsize * vals.shape[1]))
    plt.title(title)
    plt.imshow(vals, origin='lower', cmap='inferno', interpolation='nearest', vmin=0.0)

def inspect_one(feed_dict={}, idx=0, full_range=False):
    vals = tf_util.safe_run(sess, {
        'example_id': example_id,
        'voxels': voxels,
        'words':words,
        'misty_location': misty_location,
        'candidates_mask': candidates_mask,
        'match_probs':match_probs,
        'filter_probs': filter_probs,
        'filtered_mask': filtered_mask,
        'none_prob': none_prob,
        'output_block_probs':output_block_probs,
        'error': example_errors
        }, feed_dict)
    vals['words'] = tf_util.sparse_to_list(vals['words'])
    # Limit ourselves to the first example in the batch
    for k in vals:
        vals[k] = vals[k][idx]

    print('Error is', vals['error'])
    print('Sentence is', [VOCAB[i] for i in vals['words']])
    print('Location is', vals['misty_location'][0], vals['misty_location'][1])

    draw_voxels(vals['voxels'], vals['misty_location'])

    # None probs
    words_len_val = len(vals['words'])
    print('None probs:', vals['none_prob'][:words_len_val])

    # Per-word match probability
    for i, word_idx in enumerate(vals['words']):
        word = VOCAB[word_idx]
        action_prob = 1. - vals['none_prob'][i]
        if action_prob < 0.3:
            continue
        match_probs_val = vals['match_probs'][i].reshape((VOXEL_VALS_SIZE,)*2)
        if not full_range:
            match_probs_val /= np.max(match_probs_val)
        draw_heatmap('match_probs/{:02}/{}'.format(i, word),
            match_probs_val,
            vals['voxels'], vals['misty_location']
            )

    # Display "key" filters that are in-use
    for i, word_idx in enumerate(vals['words']):
        word = VOCAB[word_idx]
        action_prob = 1. - vals['none_prob'][i]
        if action_prob > 0.3:
            draw_filter('filter/{:02}/{}'.format(i, word),
                vals['filter_probs'][i])
            mask_clipped = vals['filtered_mask'][i] * vals['candidates_mask']
            mask_clipped /= np.sum(mask_clipped)
            draw_heatmap('mask/{:02}/{}_clipped'.format(i, word),
                mask_clipped,
                vals['voxels'], vals['misty_location']
                )

    # This is not a probability distribution, so normalize it
    # matching_blocks_val = vals['matching_blocks'].reshape((VOXEL_VALS_SIZE,)*2)
    # matching_blocks_val /= np.max(matching_blocks_val)
    # draw_heatmap('matching_blocks', matching_blocks_val, vals['voxels'], vals['misty_location'])

    output_block_probs_val = vals['output_block_probs'].reshape((OUTPUT_SIDE,)*2)
    if not full_range:
        output_block_probs_val /= np.max(output_block_probs_val)
    draw_heatmap(
        'output_block_probs',
        output_block_probs_val,
        vals['voxels'], vals['misty_location']
        )

# %%

varclear()

print(eval_model())

# preset_word_embeddings()
# preset_block_embeddings(norm=100.)

eval_model()

#%%

# tf.initialize_variables([x for x in tf.all_variables() if 'angular_probs_9' in x.name]).run()

inspect_one()

# tf.initialize_variables([block_embedding, global_step]).run()

# %%

set_lr(0.5)
set_emb_lr(0.05)

i = global_step.eval()
while True:
    # preset_word_embeddings()
    # preset_block_embeddings()

    if i % 100 == 0:
        print(i, eval_model(), eval_model(test_feeds), sess.run(match_probs_gap, train_feeds[0]))

    sess.run(train_op)
    i += 1

# %%
set_lr(0.05)
set_emb_lr(0.05)

i = global_step.eval()
while True:
    # preset_word_embeddings()
    # preset_block_embeddings()

    if i % 100 == 0:
        print(i, eval_model(), eval_model(test_feeds), sess.run(match_probs_gap, train_feeds[0]))

    sess.run(train_op)
    i += 1

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
