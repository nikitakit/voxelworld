# %cd ~/dev/mctest/toy2d

# import xdbg

# %%
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# %%

from gen_examples import uncache_example

# %%

import tensorflow as tf
import pandas as pd
import numpy as np
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

define('csv', default=None,
    help='path to csv file with responses', type=str)

define('out', default='responses.tfrecords',
    help='file to write responses to', type=str)

define('ptest', default=10,
    help='percent of records to allocate to test set (approximate)', type=str)

tornado.options.parse_command_line()

if options.csv is None:
    raise ValueError("Need a csv option")

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

# %%

def main(csvfile, outfile, ptest):
    df = pd.read_csv(csvfile)
    writer = tf.python_io.TFRecordWriter(outfile)
    vocabulary = []

    test_prob = ptest / 100.
    if test_prob > 0:
        test_writer = tf.python_io.TFRecordWriter(outfile.replace('.tfrecords', '.test.tfrecords'))

    df.rename(columns={'Answer.field1': 'text', 'Input.image_url':'image_url'}, inplace=True)

    for row in df.itertuples():
        example_id = os.path.splitext(os.path.basename(row.image_url))[0]
        tokens = word_tokenize(row.text)
        voxels = uncache_example(example_id)
        candidates_mask = (voxels == 0)

        for word in tokens:
            if word not in vocabulary:
                vocabulary.append(word)
        token_idxs = [vocabulary.index(word) for word in tokens]

        example = get_example(example_id, token_idxs, voxels, candidates_mask)

        if test_prob and random.random() < test_prob:
            test_writer.write(example.SerializeToString())
        else:
            writer.write(example.SerializeToString())

    vocab_file = outfile.replace('tfrecords', 'vocab')
    with open(vocab_file, 'w') as f:
        for word in vocabulary:
            f.write(word + '\n')

if __name__ == '__main__':
    main(options.csv, options.out, options.ptest)
