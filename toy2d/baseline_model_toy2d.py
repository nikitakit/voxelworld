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
0.7 S -> CLAUSE
0.2 S -> BETWEEN
0.1 S -> CLAUSE and CLAUSE
0.25 CLAUSE -> ABOVE
0.25 CLAUSE -> BELOW
0.25 CLAUSE -> LEFT
0.25 CLAUSE -> RIGHT
1.0 ABOVE -> above the @block | above_far matching_distractors
1.0 BELOW -> below the @block | below_far matching_distractors
1.0 LEFT -> left of the @block | left_far matching_distractors
1.0 RIGHT -> right of the @block | right_far matching_distractors
1.0 BETWEEN -> between the @block and the @block | between matching_distractors
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
initializer = tf.random_uniform_initializer(-sqrt3 / 100., sqrt3 / 100.)

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

voxel_vals = tf.nn.embedding_lookup(block_embedding, voxels)

VOXEL_VALS_SIZE = 19

# %%

ones_like_words = tf.SparseTensor(words.indices, tf.ones_like(words.values), words.shape)
words_len = tf.stop_gradient(tf.sparse_reduce_sum(ones_like_words, 1))
words_len.set_shape([batch_size])

words_dense = tf.sparse_tensor_to_dense(words)
words_dense.set_shape([batch_size, None])

words_dense

word_vals = tf.nn.embedding_lookup(word_embedding, words_dense)

# %% run a BiLSTM over the sentences

dense_length = tf.shape(words_dense)[1]

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
    _, lstm_outputs = tf.nn.bidirectional_dynamic_rnn(
        fw_cell, bw_cell,
        word_vals,
        sequence_length=words_len,
        dtype=tf.float32,
        scope=scope
        )

sentence_vals = tf.concat(1, [lstm_outputs[0][0], lstm_outputs[0][1], lstm_outputs[1][0], lstm_outputs[1][1]])

DIM_SENTENCE_VALS = int(sentence_vals.get_shape()[1])

#%% filter 1

FILTER1_SIZE = 5
DIM_LAYER1 = DIM_SENTENCE_VALS

with tf.device("/cpu:0"):
    filter1_bank = tf.get_variable("filter1_bank_2", [3, 3, block_emb_size, DIM_LAYER1],
        initializer=initializer, trainable=True)

layer1 = tf.nn.conv2d(
    voxel_vals,
    filter1_bank,
    strides=[1,1,1,1],
    padding="SAME"
)

layer1_s = layer1
layer1_s
# %%

OUTPUT_SIDE = VOXEL_VALS_SIZE
output_block_logits = tf.reduce_sum(tf.expand_dims(tf.expand_dims(sentence_vals, 1), 2) * layer1_s, 3)

# %%

output_block_logits_flat = tf.reshape(output_block_logits, [-1, OUTPUT_SIDE**2])
output_block_logits_flat -= 1e2 * (1. - tf.to_float(tf.reshape(candidates_mask, [-1, OUTPUT_SIDE**2])))

# for debug only
output_block_probs = tf.reshape(tf.nn.softmax(output_block_logits_flat), [-1, OUTPUT_SIDE, OUTPUT_SIDE])

misty_idxs = (misty_location[:,0] - (19 - OUTPUT_SIDE)//2) * OUTPUT_SIDE + (misty_location[:,1] - (19 - OUTPUT_SIDE)//2)

example_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(output_block_logits_flat, misty_idxs)

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

inspect_one()

# tf.initialize_variables([block_embedding, global_step]).run()

# %%

set_lr(0.5)
set_emb_lr(0.05)

i = global_step.eval()
while True:
    # preset_word_embeddings()
    # preset_block_embeddings()

    if i % 400 == 0:
        print(i, eval_model(), eval_model(test_feeds))

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
