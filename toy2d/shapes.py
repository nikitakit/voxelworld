import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# %%

from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image, ImageOps

# %matplotlib inline

# %%

SHAPES_DIR = os.path.join(os.path.dirname(__file__), "shapes")

# %%

shape_names = []
shape_images = []
inverted_shape_images = []

for filename in os.listdir(SHAPES_DIR):
    if not os.path.isfile(os.path.join(SHAPES_DIR, filename)):
        continue

    shape_img = Image.open(os.path.join(SHAPES_DIR, filename))
    shape_images.append(shape_img)

    r,g,b,a = shape_img.split()
    inverted_shape_img = ImageOps.invert(Image.merge('RGB', (r,g,b)))
    inverted_shape_img = Image.merge('RGBA', inverted_shape_img.split() + (a,))
    inverted_shape_images.append(inverted_shape_img)

    shape_names.append(os.path.splitext(filename)[0])

# %%
def draw_shapes(idxs, background=None, faded=False):
    img = Image.new(shape_images[0].mode, np.array(shape_images[0].size) * list(idxs.shape))
    subshape_mask = None
    if faded:
        if faded == True:
            faded = 125
        else:
            faded = int(faded)
        subshape_mask = Image.new('L', shape_images[0].size, (faded))

    for x in range(idxs.shape[0]):
        for y in range(idxs.shape[1]):
            if idxs[x,y] == 0:
                continue
            elif idxs[x,y] == -1:
                img.paste(shape_images[0], box=(32*x,32*y), mask=subshape_mask)
            else:
                img.paste(shape_images[idxs[x,y]], box=(32*x,32*y), mask=subshape_mask)

    bg = Image.new("RGB", img.size, (0, 0, 0))

    if background is not None:
        assert(background.shape == idxs.shape)
        background_colors = np.asarray(plt.cm.inferno(background) * 255, dtype=np.uint8)
        for x in range(idxs.shape[0]):
            for y in range(idxs.shape[1]):
                bg.paste(tuple(background_colors[x,y,:]), (32*x, 32*y, 32*x+32, 32*y+32))
                if background[x,y] > 0.6:
                    if idxs[x,y] == 0:
                        continue
                    elif idxs[x,y] == -1:
                        img.paste(inverted_shape_images[0], box=(32*x,32*y), mask=subshape_mask)
                    else:
                        img.paste(inverted_shape_images[idxs[x,y]], box=(32*x,32*y), mask=subshape_mask)

    bg.paste(img, mask=img.split()[3])
    return bg

# %%

def draw_voxels(voxels, misty_location=None):
    bg = None
    if misty_location is not None:
        bg = np.zeros_like(voxels, dtype=np.float)
        voxels[misty_location[0], misty_location[1]] = -1
        bg[misty_location[0], misty_location[1]] = 1.0
    plt.figure(figsize=(8,8))
    plt.imshow(draw_shapes(voxels, bg))

def draw_heatmap(tag, heatmap, voxels=None, misty_location=None):
    plt.figure(figsize=(8,8))
    plt.title(tag)
    if voxels is None:
        plt.xticks(np.arange(19), range(19))
        plt.yticks(np.arange(19), range(19))
        plt.grid(color='green')
        plt.imshow(heatmap, vmin=0, vmax=1.0, cmap='inferno', interpolation='none')
    else:
        if misty_location is not None:
            voxels = voxels.copy()
            voxels[misty_location[0], misty_location[1]] = -1
        plt.imshow(draw_shapes(voxels, heatmap, faded=True))


# %%
# idxs = np.arange(27, dtype=int).reshape(9,3)
# idxs[0,0] = -1

# draw_shapes(idxs, np.asarray(idxs==-1, dtype=float))
