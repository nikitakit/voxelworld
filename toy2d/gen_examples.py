# %cd ~/dev/mctest/toy2d
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# %%

import shapes
import numpy as np
import hashlib
import pathlib
import pickle


from PIL import Image

shapes.shape_images[0] = Image.open('shapes/misc/star_color.png').convert('RGBA')

# import xdbg

CSV_DIR = pathlib.Path.cwd() / 'scene_batches'
DATA_DIR = pathlib.Path.cwd() / 'scene_data'
IMAGES_DIR = pathlib.Path.cwd() / 'scene_images'
IMAGES_SUBDIR = 'v1'

UNDERLYING_SIZE = 19 + 18

def gen_example(air_probability=0.7):
    voxels = np.random.randint(1, len(shapes.shape_names),
        size=(UNDERLYING_SIZE, UNDERLYING_SIZE))
    air_mask = np.random.random(voxels.shape) < air_probability

    voxels = voxels * (1 - air_mask)
    return voxels

ALPHABET = 'abcdefgijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
def hash_example(voxels):
    voxels_str = ''.join([ALPHABET[i] for i in voxels.flatten()])
    return hashlib.sha1(voxels_str.encode()).hexdigest()

def example_to_img(voxels):
    voxels_copy = voxels[9:-9,9:-9].copy()
    voxels_copy[19//2, 19//2] = -1 # put a star here
    return shapes.draw_shapes(voxels_copy)

def cache_example(voxels, h):
    np.save(str(DATA_DIR / h), voxels)

def uncache_example(h):
    res = np.load(str(DATA_DIR / h) + '.npy')
    assert hash_example(res) == h
    return res

def generate_batch(num):
    batch_name = None
    csv_text = "image_url\n"

    for i in range(num):
        voxels = gen_example()
        h = hash_example(voxels)
        if batch_name is None:
            batch_name = h

        img = example_to_img(voxels)
        img_name = (h + '.png')
        img.save(IMAGES_DIR / IMAGES_SUBDIR / img_name)

        csv_text += IMAGES_SUBDIR + '/' + img_name + '\n'
        cache_example(voxels, h)

    csv_name = 'batch_{}.csv'.format(batch_name)

    with (CSV_DIR / csv_name).open('w') as f:
        f.write(csv_text)

    print("Wrote:", csv_name)

if __name__ == '__main__':
    generate_batch(10)
