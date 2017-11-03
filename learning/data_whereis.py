"""
This file contains the data processing pipeline for our experiments.

DataReader reads data from disk, which is what we use for our final experiments.
RandomRoomData instead generates synthetic scenes and synthetic language
descriptions to go along with them. The scenes in our synthetic Minecraft dataset
were made using this generator (though the placement of Misty was resampled
externally, and the synthetic language was replaced by human annotations).
RandomRoomData is also what we use to train our contextual block embeddings.
"""
# %cd ~/dev/mctest/toy2d
# import xdbg
#%%
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import tensorflow as tf
import threading
import random

import numpy as np
import scipy as sp
import scipy.signal

from util import tf_util
import functools
import asyncio

try:
    from tensorflow.python.training.input import _serialize_sparse_tensors as _store_sparse_tensors
    from tensorflow.python.training.input import _deserialize_sparse_tensors as _restore_sparse_tensors
except ImportError:
    from tensorflow.python.training.input import _store_sparse_tensors
    from tensorflow.python.training.input import _restore_sparse_tensors
from tensorflow.python.training.input import _shapes
from tensorflow.python.training.input import _dtypes

DATA_FORMAT = {
    'words': (tf.int64, None),
    'voxels': (tf.int64, (19, 19, 19)),
    'candidates_mask': (tf.bool, (19, 19, 19)),
    'misty_location': (tf.int64, (3,)),
    'example_id': (tf.string, ()),
}
DATA_KEY_ORDER = ['voxels', 'words', 'candidates_mask', 'misty_location', 'example_id']
EMBEDDINGS_FILENAME = os.path.expanduser("~/data/syntacticEmbeddings/skipdep_embeddings.txt")


DATA_FORMAT_UNSUPERVISED = {
    'voxels': (tf.int64, (5, 5, 5)),
    'example_id': (tf.string, ()),
}
DATA_KEY_ORDER_UNSUPERVISED = ['voxels', 'example_id']

class DataReader:
    def __init__(self, filename):
        self.filename = os.path.abspath(filename)
        self.dev_filename = self.filename.replace('.tfrecords', '.dev.tfrecords')
        self.test_filename = self.filename.replace('.tfrecords', '.test.tfrecords')
        if not os.path.isfile(self.dev_filename):
            self.dev_filename = None
        if not os.path.isfile(self.test_filename):
            self.test_filename = None
        self._dev_inputs = None
        self._test_inputs = None
        self.vocab = None
        self.max_blocks = 2 ** 12
        self.lowercase_words = True
        self.vocab_mapping = None # only used if self.lowercase_words is True

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
          serialized_example,
          {
              'words': tf.VarLenFeature(dtype=tf.int64),
              'voxels': tf.FixedLenFeature(dtype=tf.int64, shape=(19+18, 19+18, 19+18)),
              'candidates_mask': tf.FixedLenFeature(dtype=tf.int64, shape=(19+18, 19+18, 19+18)),
              'example_id': tf.FixedLenFeature(dtype=tf.string, shape=()),
          }
        )
        features['candidates_mask'] = tf.cast(features['candidates_mask'], tf.bool)
        return features

    def get_inputs(self, batch_size=1, capacity_num=100):
        global DATA_KEY_ORDER

        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer([self.filename])
            inputs_uncropped = self.read_and_decode(filename_queue)

            # Randomly select a 19x19 crop
            inputs = {
                'words': inputs_uncropped['words'],
                'example_id': inputs_uncropped['example_id']
            }

            if self.lowercase_words:
                inputs['words'] = self.to_lowercase(inputs['words'])

            crop_origin = tf.random_uniform((3,), minval=0, maxval=19, dtype=tf.int64)

            inputs['voxels'] = tf.slice(inputs_uncropped['voxels'], crop_origin, [19, 19, 19])
            inputs['candidates_mask'] = tf.slice(inputs_uncropped['candidates_mask'], crop_origin, [19, 19, 19])

            inputs['misty_location'] = tf.constant([18, 18, 18], dtype=tf.int64) - crop_origin

            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            inputs = tf.train.shuffle_batch(
                inputs, batch_size=batch_size, num_threads=2,
                capacity=capacity_num + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=capacity_num
            )

            res = tuple(inputs[key] for key in DATA_KEY_ORDER)

        if self.dev_filename is not None:
            self._prepare_dev_queue(res, batch_size=batch_size)

        if self.test_filename is not None:
            self._prepare_test_queue(res, batch_size=batch_size)

        return res

    def get_dev_feeds(self, sess=None):
        if self._dev_inputs is None:
            raise Exception("Must have dev filename and run get_inputs first")
        return self._get_feeds(self._dev_inputs, sess)

    def get_test_feeds(self, sess=None):
        if self._test_inputs is None:
            raise Exception("Must have test filename and run get_inputs first")
        return self._get_feeds(self._test_inputs, sess)

    def _get_feeds(self, inputs, sess=None):
        res = []
        while True:
            try:
                if sess is not None:
                    feed_dict = sess.run(inputs)
                else:
                    feed_dict = tf.get_default_session().run(inputs)
                res.append(feed_dict)
            except tf.errors.OutOfRangeError:
                return res

    def _prepare_dev_queue(self, placeholders, batch_size=1):
        self._dev_inputs = self._prepare_queue(placeholders, batch_size,
            filenames=[self.dev_filename], scope="input_dev")

    def _prepare_test_queue(self, placeholders, batch_size=1):
        self._test_inputs = self._prepare_queue(placeholders, batch_size,
            filenames=[self.test_filename], scope="input_test")

    def _prepare_queue(self, placeholders, batch_size, filenames, scope):
        global DATA_KEY_ORDER

        with tf.name_scope(scope):
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, capacity=1)
            inputs_uncropped = self.read_and_decode(filename_queue)

            # Select a 19x19 crop, but make sure to not crop away any candidates!
            inputs = {
                'words': inputs_uncropped['words'],
                'example_id': inputs_uncropped['example_id']
            }

            if self.lowercase_words:
                inputs['words'] = self.to_lowercase(inputs['words'])

            voxels_bool = tf.to_int32(tf.not_equal(inputs_uncropped['voxels'], 0))

            crop_x = tf.argmax(tf.reduce_max(voxels_bool, [1,2]), 0)
            crop_y = tf.argmax(tf.reduce_max(voxels_bool, [0,2]), 0)
            crop_z = tf.argmax(tf.reduce_max(voxels_bool, [0,1]), 0)

            crop_origin = tf.to_int64([crop_x, crop_y, crop_z]) - 2

            inputs['voxels'] = tf.slice(inputs_uncropped['voxels'], crop_origin, [19, 19, 19])
            inputs['candidates_mask'] = tf.slice(inputs_uncropped['candidates_mask'], crop_origin, [19, 19, 19])

            inputs['misty_location'] = tf.constant([18, 18, 18], dtype=tf.int64) - crop_origin

            assert_good_crop = tf.assert_equal(
                tf.reduce_sum(tf.to_int32(inputs_uncropped['candidates_mask'])),
                tf.reduce_sum(tf.to_int32(inputs['candidates_mask'])),
                message='Attempt to crop away candidate voxels')

            with tf.control_dependencies([assert_good_crop]):
                inputs['misty_location'] = tf.identity(inputs['misty_location'])

            # Batch the inputs using a queue
            inputs = tf.train.batch(
                inputs, batch_size=batch_size, num_threads=1,
                capacity=1,
            )

            return dict(zip(placeholders, [inputs[key] for key in DATA_KEY_ORDER]))

    def get_vocab(self):
        if self.vocab is not None:
            return self.vocab

        filename = self.filename.replace('.tfrecords', '.vocab')
        vocab = []
        with open(filename, encoding='UTF-8') as f:
            for line in f.readlines():
                vocab.append(line.strip())

        if self.lowercase_words:
            self.vocab_truecase = vocab
            self.vocab = sorted(set([x.lower() for x in vocab]))
        else:
            self.vocab = vocab

        return self.vocab

    def to_lowercase(self, words):
        assert self.lowercase_words

        self.get_vocab()
        if self.vocab_mapping is None:
            mapping = [self.vocab.index(word.lower()) for word in self.vocab_truecase]
            self.vocab_mapping = tf.constant(mapping, dtype=tf.int64)

        return tf.SparseTensor(words.indices, tf.gather(self.vocab_mapping, words.values), words.shape)

    def get_vocab_embeddings(self):
        global EMBEDDINGS_FILENAME
        vocab = self.get_vocab()

        with open(EMBEDDINGS_FILENAME, encoding='UTF-8') as f:
            # Assume that the first line is the UNK token
            _, emb = f.readline().split('\t')
            unk_embedding = np.array(emb.split(), dtype=float)
            embeddings = np.tile(unk_embedding, len(vocab)).reshape((len(vocab), -1))

            for line in f.readlines():
                word, emb = line.split('\t')
                if word not in vocab:
                    continue
                idx = vocab.index(word)
                embeddings[idx, :] = emb.split()

        return vocab, embeddings

    def create_threads(self, sess, *args, **kwargs):
        # No special threads needed to service this class, only the queue
        # runner threads
        # However, we do need to initialize the epoch counter variables
        sess.run(tf.initialize_local_variables())
        return []

class DataGenerator:
    def __init__(self, capacity, unsupervised=False):
        global DATA_FORMAT, DATA_KEY_ORDER
        global DATA_FORMAT_UNSUPERVISED, DATA_KEY_ORDER_UNSUPERVISED
        if unsupervised:
            self.data_format = DATA_FORMAT_UNSUPERVISED
            self.data_key_order = DATA_KEY_ORDER_UNSUPERVISED
        else:
            self.data_format = DATA_FORMAT
            self.data_key_order = DATA_KEY_ORDER

        self.names = []
        self.placeholders = []
        for name, (dtype, shape) in self.data_format.items():
            if shape is None:
                self.placeholders.append(tf.sparse_placeholder(dtype, name='{}_placeholder'.format(name)))
            else:
                self.placeholders.append(tf.placeholder(dtype, shape=shape, name='{}_placeholder'.format(name)))
            self.names.append(name)

        tensor_list, self.sparse_info = _store_sparse_tensors(self.placeholders[:], False)
        shapes = _shapes([tensor_list], None, False)
        dtypes = _dtypes([tensor_list])

        self.example_queue = tf.FIFOQueue(capacity, dtypes, shapes, names=self.names)
        self.enqueue_op = self.example_queue.enqueue(dict(zip(self.names, tensor_list)))

        self._test_inputs = None

    def dequeue_many(self, n=1):
        inputs = self.example_queue.dequeue_many(n)
        inputs_list = [inputs[name] for name in self.names]
        inputs_list = _restore_sparse_tensors(inputs_list, self.sparse_info)
        inputs = dict(zip(self.names, inputs_list))
        for name, (dtype, shape) in self.data_format.items():
            if shape is not None:
                inputs[name].set_shape([1] + list(shape))

        return inputs

    def prepare(self):
        pass

    async def example_gen(self):
        raise NotImplementedError()

    async def feed_queue(self, sess, coord):
        while True:
            if coord.should_stop():
                return

            examples = await self.example_gen()
            for example in examples:
                feed_dict = {}
                for i, name in enumerate(self.names):
                    if self.data_format[name][1] is None:
                        val = example[name]
                        feed_dict[self.placeholders[i]] = tf.SparseTensorValue(indices=np.arange(len(val)).reshape((-1, 1)), values=val, shape=(len(val),))
                    else:
                        feed_dict[self.placeholders[i]] = example[name]
                sess.run(self.enqueue_op, feed_dict)

    def get_inputs(self, batch_size=1, capacity_num=100):
        with tf.name_scope('input'):
            inputs = self.dequeue_many()

            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            inputs = tf.train.shuffle_batch(
                inputs, batch_size=batch_size, num_threads=1,
                capacity=capacity_num + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=capacity_num,
                enqueue_many=True
            )
            res = tuple(inputs[key] for key in self.data_key_order)

        self._test_inputs = dict(zip(res, res))

        return res

    def get_dev_feeds(self, sess=None):
        # Data is randomly generated, so use the same code for dev and test
        return self.get_test_feeds(sess)

    def get_test_feeds(self, sess=None):
        if self._test_inputs is None:
            raise Exception("Must have run get_inputs first")
        res = []
        for _ in range(50):
            try:
                if sess is not None:
                    feed_dict = sess.run(self._test_inputs)
                else:
                    feed_dict = tf.get_default_session().run(self._test_inputs)
                res.append(feed_dict)
            except tf.errors.OutOfRangeError:
                return res
        return res

    def get_vocab(self):
        raise NotImplementedError()

    def get_vocab_embeddings(self):
        global EMBEDDINGS_FILENAME
        vocab = self.get_vocab()

        with open(EMBEDDINGS_FILENAME, encoding='UTF-8') as f:
            # Assume that the first line is the UNK token
            _, emb = f.readline().split('\t')
            unk_embedding = np.array(emb.split(), dtype=float)
            embeddings = np.tile(unk_embedding, len(vocab)).reshape((len(vocab), -1))

            for line in f.readlines():
                word, emb = line.split('\t')
                if word not in vocab:
                    continue
                idx = vocab.index(word)
                embeddings[idx, :] = emb.split()

        return vocab, embeddings

    def _run(self, sess, coord):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.prepare()
        loop.run_until_complete(self.feed_queue(sess, coord))

    def create_threads(self, sess, coord, daemon=True, start=False):
        res = [threading.Thread(target=self._run, args=(sess, coord,))]
        for t in res:
            if daemon:
                t.daemon = True
            if start:
                t.start()
        return res

class SampleRejected(Exception):
    pass

class RandomRoomData(DataGenerator):
    FUNCTION_WORDS = "of the right left above below in front behind near".split()
    WALL_VOXEL = 1 # Stone
    FLOOR_VOXEL = 7 # Bedrock
    CEILING_VOXEL = 4 # Cobblestone

    def __init__(self, capacity=100, strict=False, unsupervised=False):
        super().__init__(capacity, unsupervised)
        self.max_blocks = 256
        self.strict = strict
        self.unsupervised = unsupervised

        self.vocab = set(self.FUNCTION_WORDS)

        for k in dir(self):
            if k.startswith('ground_') or k.startswith('nearwall_') or k.startswith('inwall_'):
                self.vocab |= set(k.split('_')[1:])

        self.vocab = list(self.vocab)

        self.objects_ground = []
        self.objects_nearwall = []
        self.objects_inwall = []
        self.door_fn = None
        for k in dir(self):
            if k.startswith('ground_'):
                for word in k.split('_')[1:]:
                    self.objects_ground.append(functools.partial(
                        getattr(self, k),
                        self.vocab.index(word)))
            elif k.startswith('nearwall_'):
                for word in k.split('_')[1:]:
                    self.objects_nearwall.append(functools.partial(
                        getattr(self, k),
                        self.vocab.index(word)))
            elif k.startswith('inwall_'):
                for word in k.split('_')[1:]:
                    self.objects_inwall.append(functools.partial(
                        getattr(self, k),
                        self.vocab.index(word)))

                    # Every room needs a door
                    if word == 'door':
                        self.door_fn = self.objects_inwall[-1]

        if self.unsupervised:
            self.class_counts = np.zeros(256, dtype=float)
            self.class_counts[0] = float('inf') # Never sample air!!!

    def prepare(self):
        pass

    async def example_gen(self):
        if self.unsupervised:
            example_gen_impl = self.example_gen_unsupervised
        else:
            example_gen_impl = self.example_gen_impl

        if self.strict:
            return example_gen_impl()

        max_iterations = 100
        for iteration in range(max_iterations):
            try:
                return example_gen_impl()
            except SampleRejected:
                if iteration == max_iterations - 1:
                    raise
                else:
                    continue

    # Utilities for placing objects

    def place_ground_block(self, word_id, block_id, voxels, landmarks):
        possible_locations = np.nonzero(voxels[:,0,:] == 0)
        if len(possible_locations[0]) == 0:
            raise SampleRejected()
        idx = np.random.choice(len(possible_locations[0]))
        coords = (possible_locations[0][idx], 0, possible_locations[1][idx])

        voxels[coords] = block_id
        landmarks[coords] = word_id

    def place_ground_linear(self, word_id, block_ids, axis, voxels, landmarks):
        if len(block_ids) == 1:
            return self.place_ground_block(word_id, block_ids[0], voxels, landmarks)

        if axis == 1: # y-axis
            possible_locations_mask = (voxels[:,0,:] == 0)
        elif axis == 0:
            possible_locations_mask = np.ones_like(voxels[:,0,:])
            padding = len(block_ids) - 1
            for i in range(len(block_ids)):
                possible_locations_mask[:(-padding or voxels.shape[0]),:] &= (voxels[i:(-padding+i or voxels.shape[0]),0,:] == 0)
            possible_locations_mask[(-padding or voxels.shape[0]):,:] = False
        elif axis == 2:
            possible_locations_mask = np.ones_like(voxels[:,0,:])
            padding = len(block_ids) - 1
            for i in range(len(block_ids)):
                possible_locations_mask[:,:(-padding or voxels.shape[1])] &= (voxels[:,0,i:(-padding+i or voxels.shape[1])] == 0)
            possible_locations_mask[:,(-padding or voxels.shape[0]):] = False

        possible_locations = np.nonzero(possible_locations_mask)
        if len(possible_locations[0]) == 0:
            raise SampleRejected()
        idx = np.random.choice(len(possible_locations[0]))
        xx, yy, zz = (possible_locations[0][idx], 0, possible_locations[1][idx])

        ax = int(axis == 0)
        ay = int(axis == 1)
        az = int(axis == 2)

        for i in range(len(block_ids)):
            voxels[xx + ax*i, yy + ay*i, zz + az*i] = block_ids[i]
            landmarks[xx + ax*i, yy + ay*i, zz + az*i] = word_id

    def place_in_wall(self, word_id,
            north, east, south, west,
            voxels, landmarks, min_height=1, max_height=None):
        wall_occupied_positions = (
            (voxels[:,:,1] != 0) | (landmarks[:,:,0] != 0),
            np.transpose((voxels[-2,:,:] != 0) | (voxels[-1,:,:] != 0)),
            (voxels[:,:,-2] != 0) | (landmarks[:,:,-1] != 0),
            np.transpose((voxels[1,:,:] != 0) | (landmarks[0,:,:] != 0)),
        )

        # Disallow placing on the ground!
        for el in wall_occupied_positions:
            if min_height is not None:
                el[:,:min_height] = True
            if max_height is not None:
                el[:,max_height:] = True

        return self.place_wall_helper(word_id,
            north, east, south, west,
            voxels, landmarks, wall_occupied_positions)

    def place_near_wall(self, word_id,
            north, east, south, west,
            voxels, landmarks, min_height=1, max_height=None):
        wall_occupied_positions = (
            voxels[:,:,0] != 0,
            np.transpose(voxels[-1,:,:] != 0),
            voxels[:,:,-1] != 0,
            np.transpose(voxels[0,:,:] != 0),
        )

        # Disallow placing on the ground!
        for el in wall_occupied_positions:
            if min_height is not None:
                el[:,:min_height] = True
            if max_height is not None:
                el[:,max_height:] = True

        return self.place_wall_helper(word_id,
            north, east, south, west,
            voxels, landmarks, wall_occupied_positions)

    def place_wall_helper(self, word_id,
            north, east, south, west,
            voxels, landmarks, wall_occupied_positions):
        if isinstance(north, int):
            north = [[north]]
        if isinstance(east, int):
            east = [[east]]
        if isinstance(south, int):
            south = [[south]]
        if isinstance(west, int):
            west = [[west]]

        north = np.transpose(np.asarray(north, dtype=voxels.dtype))[:,::-1]
        east = np.transpose(np.asarray(east, dtype=voxels.dtype))[:,::-1]
        south = np.transpose(np.asarray(south, dtype=voxels.dtype))[::-1,::-1]
        west = np.transpose(np.asarray(west, dtype=voxels.dtype))[::-1,::-1]

        valid_start_positions = [
            sp.signal.correlate2d(np.asarray(arr, dtype=int), np.asarray(blocks != 0, dtype=int), mode='valid') == 0
            for arr, blocks in zip(wall_occupied_positions, [north, east, south, west])
        ]

        wall_p = np.array([np.sum(arr) for arr in valid_start_positions], dtype=float)
        if np.sum(wall_p) == 0:
            raise SampleRejected()

        wall_p /= np.sum(wall_p)

        wall_i = np.random.choice(4, p=wall_p)

        start_positions = np.nonzero(valid_start_positions[wall_i])
        if len(start_positions[0]) == 0:
            raise SampleRejected()
        start_position_i = np.random.choice(len(start_positions[0]))
        start_position = [el[start_position_i] for el in start_positions]

        ax, az = 0, 0
        if wall_i == 0:
            xx, yy, zz = start_position[0], start_position[1], 0
            ax = 1
        elif wall_i == 1:
            xx, yy, zz = -1, start_position[1], start_position[0]
            az = 1
        elif wall_i == 2:
            xx, yy, zz = start_position[0], start_position[1], -1
            ax = 1
        elif wall_i == 3:
            xx, yy, zz = 0, start_position[1], start_position[0]
            az = 1

        blocks = [north, east, south, west][wall_i]
        for da in range(blocks.shape[0]):
            for dy in range(blocks.shape[1]):
                coords = (xx + ax * da, yy + dy, zz + az * da)
                if blocks[da, dy] != 0:
                    voxels[coords] = blocks[da, dy]
                    landmarks[coords] = word_id

    # Objects in the scenes

    def ground_chair(self, word, voxels, landmarks):
        self.place_ground_block(word, 53, voxels, landmarks)

    def ground_bed(self, word, voxels, landmarks):
        axis = 0
        block_ids = [26 | (3 << 8), 26 | (11 << 8)]

        # TODO: different orientations
        self.place_ground_linear(word, block_ids, axis, voxels, landmarks)

    def ground_flower(self, word, voxels, landmarks):
        self.place_ground_block(word, 38, voxels, landmarks)

    def ground_furnace(self, word, voxels, landmarks):
        self.place_ground_block(word, 62, voxels, landmarks)

    def ground_trapdoor(self, word, voxels, landmarks):
        self.place_ground_block(word, 96, voxels, landmarks)

    def _disabled_ground_tnt(self, word, voxels, landmarks):
        self.place_ground_block(word, 46, voxels, landmarks)

    def ground_table(self, word, voxels, landmarks):
        self.place_ground_linear(word, [85, 72], 1, voxels, landmarks)

    def ground_pole(self, word, voxels, landmarks):
        self.place_ground_linear(word, [85] * voxels.shape[1], 1, voxels, landmarks)

    def nearwall_bookshelf(self, word, voxels, landmarks):
        bookshelf = [[47]] * np.random.randint(2, min(4, voxels.shape[1]+1))
        self.place_near_wall(word,
            bookshelf, bookshelf, bookshelf, bookshelf,
            voxels, landmarks, min_height=0, max_height=len(bookshelf))

    def nearwall_torch(self, word, voxels, landmarks):
        self.place_near_wall(word,
            50 | (3 << 8), 50 | (2 << 8), 50 | (4 << 8), 50 | (1 << 8),
            voxels, landmarks)

    def nearwall_web(self, word, voxels, landmarks):
        self.place_near_wall(word,
            30, 30, 30, 30,
            voxels, landmarks,
            min_height=-1)

    def nearwall_shelf(self, word, voxels, landmarks):
        self.place_near_wall(word,
            72, 72, 72, 72,
            voxels, landmarks, min_height=1, max_height=min(2, voxels.shape[1]))

    def nearwall_plant(self, word, voxels, landmarks):
        plant = 31 | (1 << 8)
        self.place_near_wall(word,
            [[plant],[53 | ((4 + 3) << 8)]], [[plant],[53 | ((4) << 8)]],
            [[plant],[53 | ((4 + 2) << 8)]], [[plant],[53 | ((4 + 1) << 8)]],
            voxels, landmarks, min_height=1, max_height=3)

    def inwall_window(self, word, voxels, landmarks):
        obj = [[20, 20]]
        self.place_in_wall(word,
            obj, obj, obj, obj,
            voxels, landmarks,
            min_height=2, max_height=-2)

    def inwall_door(self, word, voxels, landmarks):
        obj = np.array([[64],
                        [64]], dtype=int)
        self.place_in_wall(word,
            obj | (3 << 8), obj | (0 << 8), obj | (1 << 8), obj | (2 << 8),
            voxels, landmarks,
            min_height=1, max_height=1+2)

    def example_gen_unsupervised(self):
        (
            room_voxels,
            room_candidates_mask,
            room_misty_loc,
            room_camera_loc,
            camera_rotation,
        ) = self.example_gen_impl(early_return=True)

        example_id = "DUMMY2:9:9:9:0.75:-0.75:0"

        room_voxels = np.pad(room_voxels, [[3,3],[3,3],[3,3]], mode='constant')
        room_candidates_mask = np.pad(room_candidates_mask, [[3,3],[3,3],[3,3]], mode='constant')

        # Pick a location adjacent to a possible Misty location
        room_interest_mask = self.get_candidates_mask(room_voxels == 0, room_candidates_mask, allow_diagonals=False)

        # Now place misty
        candidates_loc = np.nonzero(room_interest_mask)
        candidates_p = -self.class_counts[room_voxels[candidates_loc] % 256]
        candidates_p = np.exp(candidates_p - np.max(candidates_p))
        candidates_p /= candidates_p.sum()

        voxel_idx = np.random.choice(len(candidates_p), p=candidates_p)
        voxel_loc = (xx, yy, zz) = np.array([candidates_loc[i][voxel_idx] for i in range(3)])

        # Now figure out a suitable location for cropping
        voxels = room_voxels[xx-2:xx+3,yy-2:yy+3,zz-2:zz+3]

        self.class_counts[voxels[2,2,2] % 256] += 1

        return [{k: v for k,v in locals().items() if k in self.data_key_order}]

    def example_gen_impl(self, early_return=False):
        # First generate a room - note that "y" is the vertical axis
        dx, dy, dz = np.random.randint(6,15), np.random.randint(3,6), np.random.randint(6,15)

        room_voxels = np.zeros((dx, dy, dz), dtype=int)
        room_landmarks = np.zeros((dx, dy, dz), dtype=int)
        room_camera_loc = np.array([dx/2., dy - 0.1, dz - 0.1])

        camera_rotation = np.array([np.arctan2(dy+1, dz), 0., 0.])

        # Add some landmarks into the scene
        object_ids = list(np.random.choice(len(self.objects_ground), size=2, replace=False))
        object_ids.extend(np.random.choice(len(self.objects_ground), size=np.random.randint(4), replace=True))
        object_fns = [self.objects_ground[i] for i in object_ids]

        object_ids = list(np.random.choice(len(self.objects_nearwall), size=1, replace=False))
        object_ids.extend(np.random.choice(len(self.objects_nearwall), size=np.random.randint(4), replace=True))
        object_fns.extend([self.objects_nearwall[i] for i in object_ids])

        object_ids = list(np.random.choice(len(self.objects_inwall), size=np.random.randint(4), replace=True))
        object_fns_inwall = [self.door_fn] + [self.objects_inwall[i] for i in object_ids]

        for fn in object_fns:
           fn(room_voxels, room_landmarks)

        # Now add walls/floor/ceiling to the room
        room_voxels = np.pad(room_voxels, [[1,1],[0,0],[1,1]], mode='constant', constant_values=self.WALL_VOXEL)
        room_voxels = np.pad(room_voxels, [[0,0],[1,0],[0,0]], mode='constant', constant_values=self.FLOOR_VOXEL)
        room_voxels = np.pad(room_voxels, [[0,0],[0,1],[0,0]], mode='constant', constant_values=self.CEILING_VOXEL)
        room_landmarks = np.pad(room_landmarks, [[1,1],[1,1],[1,1]], mode='constant')
        room_camera_loc += np.array([1,1,1])

        # Add objects that are inside the wall (e.g. doors and windows)
        for fn in object_fns_inwall:
            fn(room_voxels, room_landmarks)

        # Calculate candidates mask after walls in place, since this crops 1 voxel off the edges
        room_candidates_mask = self.get_candidates_mask(room_voxels, room_landmarks)

        # Now place misty
        candidates_loc = np.nonzero(room_candidates_mask)
        misty_idx = np.random.choice(len(candidates_loc[0]))
        room_misty_loc = (xx, yy, zz) = np.array([candidates_loc[i][misty_idx] for i in range(3)])

        # TEST: include diagonals in the candidates mask, even though Misty can't
        # be there because we don't have relative templates for those positions
        # room_candidates_mask = self.get_candidates_mask(room_voxels, room_landmarks, allow_diagonals=False)

        if early_return:
            return (
                room_voxels,
                room_candidates_mask,
                room_misty_loc,
                room_camera_loc,
                camera_rotation,
            )

        relative_templates = [
            "right of the {}",
            "left of the {}",
            "above the {}",
            "below the {}",
            "in front of the {}",
            "behind the {}",
        ]

        relative_landmarks = [
            room_landmarks[xx-1,yy,zz],
            room_landmarks[xx+1,yy,zz],
            room_landmarks[xx,yy-1,zz],
            room_landmarks[xx,yy+1,zz],
            room_landmarks[xx,yy,zz-1],
            room_landmarks[xx,yy,zz+1],
        ]
        relative_id = np.random.choice(np.nonzero(relative_landmarks)[0])
        words = self.format_words(relative_templates[relative_id], relative_landmarks[relative_id])

        # Pad in all directions with non-candidate air blocks
        room_voxels = np.pad(room_voxels, [[18,18],[18,18],[18,18]], mode='constant')
        room_landmarks = np.pad(room_landmarks, [[18,18],[18,18],[18,18]], mode='constant')
        room_candidates_mask = np.pad(room_candidates_mask, [[18,18],[18,18],[18,18]], mode='constant')
        room_misty_loc += np.array([18,18,18])
        room_camera_loc += np.array([18,18,18])

        # Now figure out a suitable location for cropping
        crop = np.array([
            np.random.randint(room_misty_loc[0] - 19 + 4, room_misty_loc[0] - 4),
            np.random.randint(room_misty_loc[1] - 19 + 4, room_misty_loc[1] - 4),
            np.random.randint(room_misty_loc[2] - 19 + 4, room_misty_loc[2] - 4),
        ])

        voxels = room_voxels[crop[0]:crop[0]+19,crop[1]:crop[1]+19,crop[2]:crop[2]+19]
        candidates_mask = room_candidates_mask[crop[0]:crop[0]+19,crop[1]:crop[1]+19,crop[2]:crop[2]+19]
        misty_location = list(room_misty_loc - crop)
        assert voxels[misty_location[0], misty_location[1], misty_location[2]] == 0
        assert candidates_mask[misty_location[0], misty_location[1], misty_location[2]]

        camera_location = room_camera_loc - crop

        example_id = "DUMMY:{}:{}:{}:{}:{}:{}".format(*camera_location, *camera_rotation)

        return [{k: v for k,v in locals().items() if k in self.data_key_order}]

    def format_words(self, template, *landmarks):
        words_str = template.format(*[self.vocab[x] for x in landmarks])
        return [self.vocab.index(x) for x in words_str.strip().split()]

    def get_candidates_mask(self, voxels, landmarks, allow_diagonals=False):
        landmarks_bool = (landmarks != 0)
        # Only consider air blocks that are adjacent to a non-air block
        if not allow_diagonals:
            has_adjacent = (
                landmarks_bool[2:,1:-1,1:-1]
                | landmarks_bool[:-2,1:-1,1:-1]
                | landmarks_bool[1:-1,2:,1:-1]
                | landmarks_bool[1:-1,:-2,1:-1]
                | landmarks_bool[1:-1,1:-1,2:]
                | landmarks_bool[1:-1,1:-1,:-2]
            )
        else:
            adjacency_filter = np.ones((3,3,3), dtype=int)
            adjacency_filter[1,1,1] = 0

            has_adjacent = sp.signal.correlate(
                np.asarray(landmarks_bool, dtype=int),
                adjacency_filter, mode='valid') != 0

        air_and_has_adjacent = np.zeros_like(voxels)
        air_and_has_adjacent[1:-1,1:-1,1:-1] = (voxels[1:-1,1:-1,1:-1] == 0) & has_adjacent

        return air_and_has_adjacent

    def get_vocab(self):
        return self.vocab

def main():
    sess = tf.InteractiveSession()
    run_options = tf.RunOptions(timeout_in_ms=1000)

    # dg = DataReader(filename='responses.tfrecords')
    dg = RandomRoomData(strict=False)
    inputs = dg.get_inputs(batch_size=1)
    vocab, embeddings = dg.get_vocab_embeddings()
    print('Vocab:', vocab)

    coord = tf.train.Coordinator()
    coord._dg_threads = dg.create_threads(sess, coord, start=True)
    coord._my_threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('requesting...')
    try:
        res = sess.run(inputs, options=run_options)
    except tf.errors.DeadlineExceededError:
        sys.exit(1)
    misty_loc = res[-2][0]
    example_id_val = res[-1][0]
    print('misty location is', misty_loc[0], misty_loc[1], misty_loc[2])
    print('misty voxel is', res[0][0, misty_loc[0], misty_loc[1], misty_loc[2]])
    print('sentence is', [vocab[i] for i in tf_util.sparse_to_list(res[1])[0]])

    voxels_val = res[0][0, :,:,:]

    from task_tools import whereis_inspection
    if isinstance(dg, DataReader):
        task = whereis_inspection.get_task_from_example_id(example_id_val).result()
        whereis_inspection.activate_task(task).result()
    else:
        task = whereis_inspection.get_task_from_values(
            res[0][0],
            [vocab[i] for i in tf_util.sparse_to_list(res[1])[0]],
            *(res[i][0] for i in range(2, len(res)))
            )

        whereis_inspection.activate_task(task, from_values=True).result()

if __name__ == '__main__':
    main()
