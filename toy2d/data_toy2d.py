import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import tensorflow as tf
import asyncio
import threading
import random
from collections import namedtuple
import re
import functools

import numpy as np

from util import tf_util
import shapes
from shapes import draw_voxels, draw_heatmap

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
    'voxels': (tf.int64, (19, 19)),
    'candidates_mask': (tf.bool, (19, 19)),
    'misty_location': (tf.int64, (2,)),
    'example_id': (tf.string, ()),
}
DATA_KEY_ORDER = ['voxels', 'words', 'candidates_mask', 'misty_location', 'example_id']
EMBEDDINGS_FILENAME = "/Users/kitaev/dev/epic/data/syntacticEmbeddings/skipdep_embeddings.txt"

class DataGenerator:
    def __init__(self, capacity):
        global DATA_FORMAT
        self.names = []
        self.placeholders = []
        for name, (dtype, shape) in DATA_FORMAT.items():
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
        global DATA_FORMAT
        inputs = self.example_queue.dequeue_many(n)
        inputs_list = [inputs[name] for name in self.names]
        inputs_list = _restore_sparse_tensors(inputs_list, self.sparse_info)
        inputs = dict(zip(self.names, inputs_list))
        for name, (dtype, shape) in DATA_FORMAT.items():
            if shape is not None:
                inputs[name].set_shape([1] + list(shape))

        return inputs

    def prepare(self):
        pass

    async def example_gen(self):
        raise NotImplementedError()

    async def feed_queue(self, sess, coord):
        global DATA_FORMAT
        while True:
            if coord.should_stop():
                return

            examples = await self.example_gen()
            for example in examples:
                feed_dict = {}
                for i, name in enumerate(self.names):
                    if DATA_FORMAT[name][1] is None:
                        val = example[name]
                        feed_dict[self.placeholders[i]] = tf.SparseTensorValue(indices=np.arange(len(val)).reshape((-1, 1)), values=val, shape=(len(val),))
                    else:
                        feed_dict[self.placeholders[i]] = example[name]
                sess.run(self.enqueue_op, feed_dict)

    def get_inputs(self, batch_size=1, capacity_num=100):
        global DATA_KEY_ORDER
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
            res = tuple(inputs[key] for key in DATA_KEY_ORDER)

        self._test_inputs = dict(zip(res, res))

        return res

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

        with open(EMBEDDINGS_FILENAME) as f:
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

class DataReader:
    def __init__(self, filename):
        self.filename = os.path.abspath(filename)
        self.test_filename = self.filename.replace('.tfrecords', '.test.tfrecords')
        if not os.path.isfile(self.test_filename):
            self.test_filename = None
        self._test_inputs = None
        self.vocab = None
        self.max_blocks = len(shapes.shape_names)

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
          serialized_example,
          {
              'words': tf.VarLenFeature(dtype=tf.int64),
              'voxels': tf.FixedLenFeature(dtype=tf.int64, shape=(19+18, 19+18)),
              'candidates_mask': tf.FixedLenFeature(dtype=tf.int64, shape=(19+18, 19+18)),
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

            crop_origin = tf.random_uniform((2,), minval=0, maxval=19, dtype=tf.int64)

            inputs['voxels'] = tf.slice(inputs_uncropped['voxels'], crop_origin, [19, 19])
            inputs['candidates_mask'] = tf.slice(inputs_uncropped['candidates_mask'], crop_origin, [19, 19])

            inputs['misty_location'] = tf.constant([18, 18], dtype=tf.int64) - crop_origin

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

        if self.test_filename is not None:
            self._prepare_test_queue(res, batch_size=batch_size)

        return res

    def get_test_feeds(self, sess=None):
        if self._test_inputs is None:
            raise Exception("Must have test filename and run get_inputs first")
        res = []
        while True:
            try:
                if sess is not None:
                    feed_dict = sess.run(self._test_inputs)
                else:
                    feed_dict = tf.get_default_session().run(self._test_inputs)
                res.append(feed_dict)
            except tf.errors.OutOfRangeError:
                return res

    def _prepare_test_queue(self, placeholders, batch_size=1):
        global DATA_KEY_ORDER

        with tf.name_scope('input_test'):
            filename_queue = tf.train.string_input_producer([self.test_filename], num_epochs=1, capacity=1)
            inputs_uncropped = self.read_and_decode(filename_queue)

            # Randomly select a 19x19 crop
            inputs = {
                'words': inputs_uncropped['words'],
                'example_id': inputs_uncropped['example_id']
            }

            crop_origin = tf.constant([9, 9], dtype=tf.int64)

            inputs['voxels'] = tf.slice(inputs_uncropped['voxels'], crop_origin, [19, 19])
            inputs['candidates_mask'] = tf.slice(inputs_uncropped['candidates_mask'], crop_origin, [19, 19])

            inputs['misty_location'] = tf.constant([18, 18], dtype=tf.int64) - crop_origin

            # Batch the inputs using a queue
            inputs = tf.train.batch(
                inputs, batch_size=10, num_threads=1,
                capacity=1,
            )

            self._test_inputs = dict(zip(placeholders, [inputs[key] for key in DATA_KEY_ORDER]))

    def get_vocab(self):
        if self.vocab is not None:
            return self.vocab

        filename = self.filename.replace('.tfrecords', '.vocab')
        vocab = []
        with open(filename) as f:
            for line in f.readlines():
                vocab.append(line.strip())

        self.vocab = vocab
        return self.vocab

    def get_vocab_embeddings(self):
        global EMBEDDINGS_FILENAME
        vocab = self.get_vocab()

        with open(EMBEDDINGS_FILENAME) as f:
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

class SampleRejected(Exception):
    pass

class GrammarGeneratedData(DataGenerator):
    """
    A data generator.
    """

    DEFAULT_GRAMMAR = """
1.0 ROOT -> S | random_fill_some
0.7 S -> CLAUSE
0.3 S -> COMPOUND_CLAUSE
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

    Rule = namedtuple('Rule', ['p', 'parent', 'children', 'functions'])

    SPECIAL_SYMBOLS = {"@block"}

    def __init__(self, capacity=100, grammar=None, strict=False):
        super().__init__(capacity)
        self.strict = strict
        self.max_blocks = len(shapes.shape_names)
        self.underlying_size = 19 + 18
        self.mask = np.ones((self.underlying_size, self.underlying_size), dtype = bool)
        self.center = np.array([self.underlying_size, self.underlying_size], dtype=int) // 2
        self.misty_mask = np.zeros_like(self.mask)
        self.misty_mask[tuple(self.center)] = True

        polarity = (self.underlying_size // 2) % 2
        for i in range(self.underlying_size):
            for j in range(self.underlying_size):
                if i % 2 == polarity and j % 2 == polarity:
                    self.mask[i,j] = False

        if grammar is not None:
            self.grammar = grammar
        else:
            grammar = self.DEFAULT_GRAMMAR

        self.nonterminal_symbols = set()
        self.rules = []
        self.terminal_symbols = set()

        for line in grammar.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match('(?P<p>.+) (?P<parent>.+) -> (?P<children>(?:.|[ ])+) \| (?P<functions>(?:.|[ ])+)', line)
            if match:
                g = match.groupdict()
            else:
                match = re.match('(?P<p>.+) (?P<parent>.+) -> (?P<children>(?:.|[ ])+)', line)
                if not match:
                    raise ValueError("Could not parse line: {}".format(line))
                g = match.groupdict()
                g['functions'] = ''

            rule = self.Rule(
                p=float(g['p']),
                parent=g['parent'],
                children=g['children'].split(),
                functions=[getattr(self, x) for x in g['functions'].split()]
            )

            self.nonterminal_symbols.add(rule.parent)
            self.rules.append(rule)

        # Verify that all symbols are valid
        for rule in self.rules:
            for symbol in rule.children:
                if symbol.startswith('@'):
                    if symbol not in self.SPECIAL_SYMBOLS:
                        raise ValueError("Invalid symbol: {}".format(symbol))
                    else:
                        continue

                if symbol.islower():
                    self.terminal_symbols.add(symbol)
                elif symbol not in self.nonterminal_symbols:
                    raise ValueError("No rule for nonterminal: {}".format(symbol))

        # Construct lookup table for rules, based on the parent symbol
        self.rules_table = {}
        for symbol in self.nonterminal_symbols:
            valid_rules = [rule for rule in self.rules if rule.parent == symbol]
            probabilities = [rule.p for rule in valid_rules]
            if not np.isclose(sum(probabilities), 1.0):
                raise ValueError("Rule probabilities for nonterminal {} do not sum to 1".format(symbol))

            self.rules_table[symbol] = (np.array(probabilities, dtype=float), valid_rules)

        self.vocab = shapes.shape_names + sorted(self.terminal_symbols)

    def prepare(self):
        pass

    def priority(value):
        def dec(fn):
            fn._priority = value
            return fn
        return dec

    @priority(0)
    def air_grid(self, reserved, remaining, voxels, unassigned):
        air_mask = (~self.mask) & unassigned
        voxels[:,:] = np.where(air_mask, np.zeros_like(voxels), voxels)
        unassigned[:,:] = np.where(air_mask, False, unassigned)

    @priority(10)
    def between(self, reserved, remaining, voxels, unassigned, offsets=[-1, 0, 1]):
        assert len(reserved) == 2

        offset = np.zeros(2, dtype=int)
        for _ in range(100):
            if unassigned[tuple(self.center + offset)] and unassigned[tuple(self.center - offset)]:
                break
            offset = np.random.choice(offsets, 2)
        else:
            raise SampleRejected("Could not find valid location for landmark")

        voxels[tuple(self.center + offset)] = reserved[0]
        voxels[tuple(self.center - offset)] = reserved[1]
        unassigned[tuple(self.center + offset)] = False
        unassigned[tuple(self.center - offset)] = False

    @priority(10)
    def between_far(self, *args):
        self.between(*args, offsets=list(range(-6,6+1)))

    def cardinal(self, reserved, remaining, voxels, unassigned, direction, distance):
        assert len(reserved) == 1

        landmark_loc = tuple(self.center + np.asarray(direction) * distance)
        if not unassigned[landmark_loc]:
            raise SampleRejected("Location for above is already occupied")

        for intermediate_distance in range(1, distance):
            intermediate_loc = tuple(self.center + np.asarray(direction) * intermediate_distance)
            if not unassigned[intermediate_loc] and voxels[intermediate_loc] != 0:
                raise SampleRejected("Distractor voxel is between target and landmark")

        for intermediate_distance in range(1, distance):
            intermediate_loc = tuple(self.center + np.asarray(direction) * intermediate_distance)
            if unassigned[intermediate_loc]:
                voxels[intermediate_loc] = 0
                unassigned[intermediate_loc] = False

        voxels[landmark_loc] = reserved[0]
        unassigned[landmark_loc] = False

    @priority(10)
    def above(self, *args):
        self.cardinal(*args, [0,1], 1)

    @priority(10)
    def below(self, *args):
        self.cardinal(*args, [0,-1], 1)

    @priority(10)
    def left(self, *args):
        self.cardinal(*args, [1,0], 1)

    @priority(10)
    def right(self, *args):
        self.cardinal(*args, [-1,0], 1)

    def gen_distance_far(self):
        return np.random.choice([1,2,3,4], p=[0.3, 0.35, 0.3, 0.05])

    @priority(10)
    def below_far(self, *args):
        self.cardinal(*args, [0,-1], self.gen_distance_far())

    @priority(10)
    def above_far(self, *args):
        self.cardinal(*args, [0,1], self.gen_distance_far())

    @priority(10)
    def left_far(self, *args):
        self.cardinal(*args, [1,0], self.gen_distance_far())

    @priority(10)
    def right_far(self, *args):
        self.cardinal(*args, [-1,0], self.gen_distance_far())

    def cardinal_compound(self, reserved, remaining, voxels, unassigned,
            direction, distance, ref_direction, ref_distance):
        assert len(reserved) == 2

        landmark_loc = tuple(self.center + np.asarray(direction) * distance)
        if not unassigned[landmark_loc]:
            raise SampleRejected("Location for above is already occupied")

        for intermediate_distance in range(1, distance):
            intermediate_loc = tuple(self.center + np.asarray(direction) * intermediate_distance)
            if not unassigned[intermediate_loc] and voxels[intermediate_loc] != 0:
                raise SampleRejected("Distractor voxel is between target and landmark")

        ref_loc = tuple(np.asarray(landmark_loc) + np.asarray(ref_direction) * ref_distance)
        if not unassigned[ref_loc]:
            raise SampleRejected("Location for above is already occupied")

        for intermediate_distance in range(1, ref_distance):
            intermediate_loc = tuple(np.asarray(landmark_loc) + np.asarray(ref_direction) * intermediate_distance)
            if not unassigned[intermediate_loc] and voxels[intermediate_loc] != 0:
                raise SampleRejected("Distractor voxel is between landmark and reference")

        for intermediate_distance in range(1, distance):
            intermediate_loc = tuple(self.center + np.asarray(direction) * intermediate_distance)
            if unassigned[intermediate_loc]:
                voxels[intermediate_loc] = 0
                unassigned[intermediate_loc] = False

        voxels[landmark_loc] = reserved[0]
        unassigned[landmark_loc] = False

        for intermediate_distance in range(1, ref_distance):
            intermediate_loc = tuple(np.asarray(landmark_loc) + np.asarray(ref_direction) * intermediate_distance)
            if unassigned[intermediate_loc]:
                voxels[intermediate_loc] = 0
                unassigned[intermediate_loc] = False

        voxels[ref_loc] = reserved[1]
        unassigned[ref_loc] = False

    @priority(10)
    def above_above(self, *args):
        self.cardinal_compound(*args, [0,1], 1, [0,1], 1)

    @priority(10)
    def above_left(self, *args):
        self.cardinal_compound(*args, [0,1], 1, [1,0], 1)

    @priority(10)
    def above_right(self, *args):
        self.cardinal_compound(*args, [0,1], 1, [-1,0], 1)

    @priority(10)
    def below_below(self, *args):
        self.cardinal_compound(*args, [0,-1], 1, [0,-1], 1)

    @priority(10)
    def below_left(self, *args):
        self.cardinal_compound(*args, [0,-1], 1, [1,0], 1)

    @priority(10)
    def below_right(self, *args):
        self.cardinal_compound(*args, [0,-1], 1, [-1,0], 1)

    @priority(10)
    def left_above(self, *args):
        self.cardinal_compound(*args, [1,0], 1, [0,1], 1)

    @priority(10)
    def left_below(self, *args):
        self.cardinal_compound(*args, [1,0], 1, [0,-1], 1)

    @priority(10)
    def left_left(self, *args):
        self.cardinal_compound(*args, [1,0], 1, [1,0], 1)

    @priority(10)
    def right_above(self, *args):
        self.cardinal_compound(*args, [-1,0], 1, [0,1], 1)

    @priority(10)
    def right_below(self, *args):
        self.cardinal_compound(*args, [-1,0], 1, [0,-1], 1)

    @priority(10)
    def right_right(self, *args):
        self.cardinal_compound(*args, [-1,0], 1, [-1,0], 1)

    @priority(90)
    def matching_distractors(self, reserved, remaining, voxels, unassigned, max_count=4):
        random_nums = np.random.random(voxels.shape)
        # make distractors within view of the target more likely
        random_nums[19//2:19//2+19,19//2:19//2+19] *= 0.2
        xs, ys = np.unravel_index(random_nums.ravel().argsort(), random_nums.shape)

        distractors_to_add = list(reserved)[::] * max_count
        for x, y in zip(xs, ys):
            if not distractors_to_add:
                break
            if not unassigned[x, y]:
                continue
            voxels[x, y] = distractors_to_add.pop()
            unassigned[x, y] = False
        else:
            raise SampleRejected("No locations found for distractors")

    @priority(90)
    def compound_distractors(self, reserved, remaining, voxels, unassigned, max_count=4):
        random_nums = np.random.random(voxels.shape)
        # make distractors within view of the target more likely
        random_nums[19//2:19//2+19,19//2:19//2+19] *= 0.2
        xs, ys = np.unravel_index(random_nums.ravel().argsort(), random_nums.shape)

        # Distractors should match the landmark, not the relative clause
        distractors_to_add = [reserved[0]] * max_count
        for x, y in zip(xs, ys):
            if not distractors_to_add:
                break
            if not unassigned[x, y]:
                continue
            voxels[x, y] = distractors_to_add.pop()
            unassigned[x, y] = False
        else:
            raise SampleRejected("No locations found for distractors")

    @priority(100)
    def random_fill_some(self, reserved, remaining, voxels, unassigned, air_probability=0.5):
        random_voxels = np.random.randint(len(remaining), size=voxels.shape)
        random_voxels = np.array([remaining[i] for i in random_voxels.flatten()]).reshape(voxels.shape)

        air_mask = np.random.random(random_voxels.shape) < air_probability

        random_voxels = np.where(air_mask, np.zeros_like(random_voxels), random_voxels)

        voxels[:,:] = np.where(unassigned, random_voxels, voxels)
        unassigned[:,:] = False

    @priority(101)
    def random_fill(self, reserved, remaining, voxels, unassigned):
        # Fill in air positions
        random_voxels = np.random.randint(len(remaining), size=voxels.shape)
        random_voxels = np.array([remaining[i] for i in random_voxels.flatten()]).reshape(voxels.shape)
        voxels[:,:] = np.where(unassigned, random_voxels, voxels)
        unassigned[:,:] = False

    async def example_gen(self):
        if self.strict:
            return await self.example_gen_impl()

        max_iterations = 100
        for iteration in range(max_iterations):
            try:
                return await self.example_gen_impl()
            except SampleRejected:
                if iteration == max_iterations - 1:
                    raise
                else:
                    continue

    async def example_gen_impl(self):
        all_blocks = list(range(1, self.max_blocks))

        voxels = np.zeros_like(self.mask, dtype=int)
        unassigned = np.zeros_like(voxels, dtype=bool)

        words = []
        queued_functions = []

        def expand(nonterminal):
            nonlocal all_blocks, voxels, unassigned
            nonlocal words, queued_functions

            p, rules = self.rules_table[nonterminal]
            i = np.random.choice(len(rules), p=p)
            rule = rules[i]

            assert rule.parent == nonterminal

            reserved = []
            for symbol in rule.children:
                if symbol in self.nonterminal_symbols:
                    expand(symbol)
                elif symbol == "@block":
                    block_id = np.random.choice(all_blocks)
                    all_blocks.remove(block_id)
                    reserved.append(block_id)
                    words.append(block_id)
                else:
                    words.append(self.vocab.index(symbol))

            for function in rule.functions:
                queued_functions.append((
                    function._priority,
                    function.__name__,
                    functools.partial(function, reserved, all_blocks, voxels, unassigned)
                    ))

        expand("ROOT")

        queued_functions.sort(key = lambda x: x[0])

        max_iterations = 10
        for iteration_num in range(max_iterations):
            voxels[:,:] = 0
            unassigned[:,:] = True
            unassigned[tuple(self.center)] = False

            try:
                for _, _, function in queued_functions:
                    function()
                break
            except SampleRejected:
                if iteration_num == (max_iterations - 1):
                    raise
                else:
                    continue

        assert not np.any(unassigned), "Found an unassigned voxel"

        candidates_mask = (voxels == 0)
        example_id = "DUMMY"

        # Now crop the voxels and candidates to 19x19
        absoffset = np.array([0,0])
        crop = np.array([
            np.random.randint(absoffset[0], 19-absoffset[0]),
            np.random.randint(absoffset[1], 19-absoffset[1])
        ])

        voxels = voxels[crop[0]:crop[0]+19,crop[1]:crop[1]+19]

        candidates_mask = candidates_mask[crop[0]:crop[0]+19,crop[1]:crop[1]+19]
        misty_mask = self.misty_mask[crop[0]:crop[0]+19,crop[1]:crop[1]+19]
        misty_location = [int(loc) for loc in np.where(misty_mask)]
        assert voxels[misty_location[0], misty_location[1]] == 0

        return [{k: v for k,v in locals().items() if k in DATA_KEY_ORDER}]

    def get_vocab(self):
        return self.vocab

def main():
    sess = tf.InteractiveSession()
    run_options = tf.RunOptions(timeout_in_ms=1000)

    dg = GrammarGeneratedData(strict=True)
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
    print('misty location is', misty_loc[0], misty_loc[1])
    print('misty voxel is', res[0][0, misty_loc[0], misty_loc[1]])
    print('sentence is', [vocab[i] for i in tf_util.sparse_to_list(res[1])[0]])

    if True:
        from matplotlib import pyplot as plt
        draw_voxels(res[0][0,:,:], misty_loc)
        plt.show()

if __name__ == '__main__':
    main()
