import tensorflow as tf
import uuid, contextlib
from tensorflow.python.ops import variable_scope as vs

###############################################################################
# DATA PROCESSING
###############################################################################

def sparse_to_list(sparse_value):
    res = [[] for _ in range(sparse_value.shape[0])]
    for (i,j), val in zip(sparse_value.indices, sparse_value.values):
        res[i].append(val)
    return res

def list_to_sparse(lst):
    # List should be 2D
    indices = []
    values = []
    shape = [len(lst),0]

    for i, row in enumerate(lst):
        for j, el in enumerate(row):
            indices.append((i,j))
            values.append(el)
            shape[1] = max(shape[1], j+1)

    return tf.SparseTensorValue(indices=indices, values=values, shape=shape)

def sparse_vectorize(lmbda):
    return lambda arg: [[lmbda(el) for el in row] for row in arg]

###############################################################################
# Layers
###############################################################################

def linear(arg, output_size, bias=True, scope=None):
    """
    Construct a linear layer with a bias term
    """

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [int(arg.get_shape()[-1]), output_size], dtype=arg.dtype)

        flat_arg = tf.reshape(arg, [-1, int(arg.get_shape()[-1])])
        res_flat = tf.matmul(flat_arg, matrix)

        res = tf.reshape(res_flat, [a.value or -1 for a in arg.get_shape()][:-1] + [output_size])

        # input_names = "abcdef"[:len(arg.get_shape()) - 1]
        # res = tf.einsum("{0}j,jk->{0}k".format(input_names), arg, matrix)
        if not bias:
            return res

        bias_term = tf.get_variable("Bias", [output_size], dtype=arg.dtype, initializer=tf.zeros_initializer)
        return res + bias_term

###############################################################################
# Variable setters
###############################################################################

def create_row_setter(variable, scope=None):
    if scope is not None:
        with tf.variable_scope(scope):
            return create_row_setter(variable, scope=None)

    row_length = int(variable.get_shape()[1])

    row_numbers = tf.placeholder(shape=(None,), dtype=tf.int64, name="row_numbers")
    row_values = tf.placeholder(shape=(None, row_length), dtype=variable.value().dtype, name="row_values")
    set_row_op = tf.scatter_update(variable, row_numbers, row_values, name="set_row_op")

    def set_row(number, values):
        set_row_op.eval({
            row_numbers: number,
            row_values: values
        })

    return set_row

def create_var_setter(variable, scope=None):
    if scope is not None:
        with tf.variable_scope(scope):
            return create_var_setter(variable, scope=None)

    var_value = tf.placeholder(shape=variable.get_shape(), dtype=variable.value().dtype, name="var_value")
    set_var_op = tf.assign(variable, var_value, name="set_var_op")

    def set_var(value):
        set_var_op.eval({
            var_value: value
        })

    return set_var

###############################################################################
# TENSORFLOW UTILITIES
###############################################################################

@contextlib.contextmanager
def new_variable_scope(name):
    """
    Like vs.variable_scope, except it picks a different name each time.
    (for use in interactive sessions)
    """
    name = name + str(uuid.uuid4())[-13:]
    with vs.variable_scope(name) as v:
        yield v

def sparse_boolean_mask(tensor, mask):
    """
    Creates a sparse tensor from masked elements of `tensor`

    Inputs:
      tensor: a 2-D tensor, [batch_size, T]
      mask: a 2-D mask, [batch_size, T]

    Output: a 2-D sparse tensor
    """
    mask_lens = tf.reduce_sum(tf.cast(mask, tf.int32), -1, keep_dims=True)
    mask_shape = tf.shape(mask)
    left_shifted_mask = tf.tile(
        tf.expand_dims(tf.range(mask_shape[1]), 0),
        [mask_shape[0], 1]
    ) < mask_lens
    return tf.SparseTensor(
        indices=tf.where(left_shifted_mask),
        values=tf.boolean_mask(tensor, mask),
        shape=tf.cast(tf.pack([mask_shape[0], tf.reduce_max(mask_lens)]), tf.int64) # For 2D only
    )

def sparse_boolean_mask_length_capped(tensor, mask, lens):
    """
    Creates a sparse tensor from masked elements of `tensor`

    Inputs:
      tensor: a 2-D tensor, [batch_size, T]
      mask: a 2-D mask, [batch_size, T]
      lens: a 1-D tensor, [batch_size]
            This additionally masks out all values in `tensor`
            where their 2-d dimension is greater than the len
            for that batch

    Output: a 2-D sparse tensor
    """
    if lens.dtype == tf.int64:
        lens = tf.cast(lens, tf.int32)

    mask_shape = tf.shape(mask)
    range_like_mask = tf.tile(
            tf.expand_dims(tf.range(mask_shape[1]), 0),
        [mask_shape[0], 1]
    )

    length_mask = range_like_mask < tf.expand_dims(lens, 1)
    mask = mask & length_mask

    mask_lens = tf.reduce_sum(tf.cast(mask, tf.int32), -1, keep_dims=True)
    left_shifted_mask = range_like_mask < mask_lens
    return tf.SparseTensor(
        indices=tf.where(left_shifted_mask),
        values=tf.boolean_mask(tensor, mask),
        shape=tf.cast(tf.pack([mask_shape[0], tf.reduce_max(mask_lens)]), tf.int64) # For 2D only
    )

def reverse_dynamic_rnn(cell_bw, inputs, sequence_length=None,
                              initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
  """
  Based on https://github.com/tensorflow/tensorflow/pull/2581
  That PR added tf.nn.bidirectional_dynamic_rnn. This is a modified version that
  implements the backwards RNN only.
  """
  if True:
    from tensorflow.python.ops.rnn import dynamic_rnn
    from tensorflow.python.framework import dtypes
    from tensorflow.python.framework import ops
    from tensorflow.python.framework import tensor_shape
    from tensorflow.python.framework import tensor_util
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import control_flow_ops
    from tensorflow.python.ops import logging_ops
    from tensorflow.python.ops import math_ops
    from tensorflow.python.ops import rnn_cell
    from tensorflow.python.ops import tensor_array_ops
    from tensorflow.python.ops import variable_scope as vs

  if not isinstance(cell_bw, rnn_cell.RNNCell):
    raise TypeError("cell_bw must be an instance of RNNCell")

  name = scope or "ReverseRNN"

  # Backward direction
  if not time_major:
    time_dim = 1
    batch_dim = 0
  else:
    time_dim = 0
    batch_dim = 1
  with vs.variable_scope(name + "_BW") as bw_scope:
    inputs_reverse = array_ops.reverse_sequence(
        inputs, sequence_length, time_dim, batch_dim)
    tmp, output_state_bw = dynamic_rnn(
        cell_bw, inputs_reverse, sequence_length, initial_state_bw, dtype,
        parallel_iterations, swap_memory, time_major, scope=bw_scope)
  output_bw = array_ops.reverse_sequence(
      tmp, sequence_length, time_dim, batch_dim)

  return output_bw, output_state_bw

###############################################################################
# MISC
###############################################################################
from copy import deepcopy
def safe_run(sess, fetches, feed_dict):
    """
    Like sess.run, but supports feeding and fetching the same tensor.
    This is useful for test-time, where you want the same codepath to run with
    either a randomly fetched example, or a manually specified one
    """
    if isinstance(fetches, dict):
        alt_fetches = {}
        known_results = {}
        for k, v in fetches.items():
            if v in feed_dict:
                try:
                    known_results[k] = deepcopy(feed_dict[v])
                except:
                    known_results[k] = feed_dict[v]
            else:
                alt_fetches[k] = v
        res = sess.run(alt_fetches, feed_dict)
        res.update(known_results)
        return res
    else:
        raise NotImplementedError("Only dict fetches supported for now")

def normalize_distribution(tensor, dim):
    tensor_sums = tf.reduce_sum(tensor, dim, keep_dims=True)
    # don't divide by zero!
    return tensor / tf.select(tensor_sums > 0,
                                tensor_sums,
                                tf.ones_like(tensor_sums))
