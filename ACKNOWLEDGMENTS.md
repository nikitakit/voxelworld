# Acknowledgments

The files/folders mentioned below are copied or derived from other projects. See the files themselves for complete copyright notices and licenses.

### voxel-ui/static/textures/pixelperfection.zip

This texture package is based on the Pixel Perfection texture pack, version 3.5, by XSSheep.

A smaller number of missing textures have been filled in from the ProgrammerArt texture pack, version 3.0, by deathcap.

We have adapted the original textures for our renderer. See the `README.txt` inside the zip archive for details.

License: CC-BY-SA

### voxel-ui/vendor/block-models

Original at https://github.com/deathcap/block-models/

Commit hash: `70e9c3becfcbf367e31e5b158ea8766c2ffafb5e`

Our version uses a different JSON format from the original, to support a greater number of mesh features.

### voxel-ui/vendor/voxel-engine-stackgl

Original at https://github.com/voxel/voxel-engine-stackgl

Commit hash: `dc213bd8914f376237cf807d1d127a10ebaed11c`

This folder was imported primarily because of transitive dependencies on other parts of voxel. It has also been slightly modified with respect to handling of options (for example, the original does not pass the `preserveDrawingBuffer` flag, which is required to take screenshots.)

### voxel-ui/vendor/voxel-mesher

Original at https://github.com/voxel/voxel-mesher

Commit hash: `fc6d48247fd9393e32db89a6f239317212fdbb9b`

Our version includes several performance fixes, which are required to handle the large set of non-cubic block models our project uses. It is also needed because of the transitive dependency on block-models.

### voxel-ui/vendor/voxel-outline

Original at https://github.com/voxel/voxel-outline

Commit hash: `2299edd018e4c7141f419a4e1b280ba202d9911f`

We have added a `selectAdjacent` option not present in the original code.

### voxel-ui

Our web application in the `voxel-ui` folder is based on the BSD-licensed `voxel-hello-world` example code.

### util/ipyloop.py

Original at https://github.com/ipython/ipykernel/issues/21

### util/transformations.py

Original at http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

We have modified it to not warn about lacking the corresponding c module.
