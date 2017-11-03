// This file wraps voxel-engine-stackgl, in that it loads the main voxel engine,
// any plugins we use, and makes sure that everything renders within the correct
// container in the DOM.
var ndarray = require('ndarray')
var createGame = require('voxel-engine-stackgl')

function chunkLoaderPlugin(game, opts) {
    var chunkConverter = require('./chunk-converter')(game);
    var chunkLoader = new ChunkLoader(game, chunkConverter);
    game.chunkLoader = chunkLoader;
    return chunkLoader;
}

chunkLoaderPlugin.pluginInfo = {
    loadAfter: ['voxel-artpacks', 'voxel-stitch']
}

module.exports = function() {
    var game = require('voxel-engine-stackgl')({
      pluginLoaders: {
          'voxel-artpacks': require('voxel-artpacks'),
          'voxel-keys': require('voxel-keys'),
          'voxel-wireframe': require('voxel-wireframe'),
          'voxel-outline': require('voxel-outline'),
          'heatmap': require('./heatmap'),
          'custom-controls': require('./custom-controls'),
          'chunk-loader': chunkLoaderPlugin,
          'task-utils': require('./task-utils')
      },
      pluginOpts: {
        'voxel-engine-stackgl': {
            generateChunks: true,
            texturePath: 'ArtPacks/ProgrammerArt/textures/blocks/'
        },
        'game-shell-fps-camera': {
            position: [-2, -3.5, -2],
            rotationY: Math.PI * 3 / 4
        },
        'voxel-artpacks': {},
        'voxel-keys': {},
        'voxel-stitch': {
            artpacks: ['static/textures/pixelperfection.zip'],
            verbose: false
        },
        'voxel-wireframe': {showWireframe: false},
        'voxel-outline': {selectAdjacent: true},
        'heatmap': {},
        'custom-controls': {},
        'chunk-loader': {},
        'task-utils': {}
      },
      container: document.getElementById('container'),

      controls: {
          discreteFire: true
      },

      worldOrigin: [0, 0, 0],
      chunkDistance: 2,
      removeDistance: 4,

      // WARNING: chunkPad=4 is the default and also the only allowed value.
      // I (Nikita) have tried and failed to find all the places in the voxel
      // code where the value of "4" is hardcoded, instead of checking the
      // option.
      // This value is hardcoded in at least:
      // * mesh-plugin.js (search for HACK)
      // * Somewhere in the solid block mesher + maybe the voxel-shader ao.vsh
      //   file
      //
      // If you plan on changing this option, be sure to check that world
      // coordinates align with block coordinates (i.e. camera at (0,0,0) is at
      // the correct corner of the block at (0,0,0))
      chunkPad: 4,

      generateVoxelChunk: function(lo, hi) {
        lo[0]-=2 // 2 is half the chunkPad
        lo[1]-=2
        lo[2]-=2
        hi[0]+=2
        hi[1]+=2
        hi[2]+=2
        var dims = [hi[2]-lo[2], hi[1]-lo[1], hi[0]-lo[0]]
        var data = ndarray(new Uint16Array(dims[2] * dims[1] * dims[0]), dims)
        for (var k = lo[2]; k < hi[2]; k++)
          for (var j = lo[1]; j < hi[1]; j++)
            for(var i = lo[0]; i < hi[0]; i++) {
              data.set(k-lo[2], j-lo[1], i-lo[0], 0)
            }
        game.chunkLoader.requestChunk(lo, dims, data)

        return data
      },

      arrayTypeSize: 2 // Use uint16 to store blocks, because uint8 is too small
    });

    // For debugging
    game.$ = require('jquery');
    window.game = game;
    window.$ = game.$;

    game.plugins.get('voxel-outline').showOutline = false;

    // Default position for the avatar
    var avatarPosition = game.controls.target().avatar.position;
    avatarPosition.x = 28;
    avatarPosition.y = 28;
    avatarPosition.z = 28;

    var avatarRotation = game.controls.target().avatar.rotation;
    avatarRotation.x = Math.PI * 1 / 4;
    avatarRotation.y = -Math.PI * 1 / 4;
    avatarRotation.z = 0;

    game.stitcher.on('updateTexture', function(){
        $('#startup-text').hide();
    });

    return game;
}

function ChunkLoader(game, chunkConverter) {
    var self = this;

    this.game = game;
    this.chunkConverter = chunkConverter;

    this.regionsToData = {};

    this.enabled = true;
    this.requestChunksFunction = null;
    // Before a requestChunksFunction is set, requests are queued up
    this.chunksToLoadLater = [];
}

ChunkLoader.prototype.regionToKey = function(position, dimensions) {
    return position.join("|") + "+" + dimensions.join("|");
}

ChunkLoader.prototype.requestChunk = function(position, dimensions, data) {
    if (!this.enabled) {
        return;
    }

    if (this.requestChunksFunction === null) {
        this.chunksToLoadLater.push([position, dimensions, data]);
        return;
    }

    this.regionsToData[this.regionToKey(position, dimensions)] = data;

    this.requestChunksFunction(position, dimensions, data);
}

ChunkLoader.prototype.onChunkRequested = function(callback) {
    this.requestChunksFunction = callback;
    if (callback != null) {
        for (var i = 0; i < this.chunksToLoadLater.length; i++) {
            this.requestChunk(this.chunksToLoadLater[i][0], this.chunksToLoadLater[i][1], this.chunksToLoadLater[i][2]);
        }
        this.chunksToLoadLater = [];
    }
}

ChunkLoader.prototype.enable = function() {
    this.enabled = true;
}

ChunkLoader.prototype.disable = function() {
    this.enabled = false;

    // empty out list of chunks to load later
    this.regionsToData = {};
    this.chunksToLoadLater = [];
}

ChunkLoader.prototype.drawRegion = function(region) {
    var game = this.game;

    var dataCandidate = this.regionsToData[this.regionToKey(region.position, region.dimensions)];
    if (dataCandidate !== undefined) {
        // We have direct access to the chunk, so we can use this faster way of writing
        // out the data
        var idx = 0;
        for (var i=0; i < region.dimensions[0]; i++) {
            for (var j=0; j < region.dimensions[1]; j++) {
                for (var k=0; k < region.dimensions[2]; k++, idx++) {
                    dataCandidate.set(i, j, k, this.chunkConverter.translateBlockID(region.voxelsU32.voxelsU32[idx]));
                }
            }
        }

        game.addChunkToNextUpdate(dataCandidate);
    } else {
        var ox = region.position[0];
        var oy = region.position[1];
        var oz = region.position[2];

        var chunkSize = game.voxels.chunkSize;
        var mask = game.voxels.chunkMask;
        var h = game.voxels.chunkPadHalf;

        function setInChunk(loopStep, c, i, j, k, dx, dy, dz, val) {
            var chunk = game.voxels.chunks[c.join('|')];
            if (!chunk) {
                game.voxels.generateChunk(c[0], c[1], c[2]);
                chunk = game.voxels.chunks[c.join('|')];
            }

            chunk.set((i & mask) + h + dx, (j & mask) + h + dy, (k & mask) + h + dz, val);
            game.addChunkToNextUpdate(chunk);

            // If this position is also included in another chunk's padding, it
            // needs to be set there, too. Otherwise there will be rendering
            // artifacts because chunk meshes need overlap to stitch up correctly.
            // This is also why we can't just call setBlock
            if (loopStep == 0) {
                if ((i & mask) < h) {
                    setInChunk(1, [c[0]-1, c[1], c[2]], i, j, k, dx+chunkSize, dy, dz, val);
                } else if ((i & mask) >= chunkSize - h) {
                    setInChunk(1, [c[0]+1, c[1], c[2]], i, j, k, dx-chunkSize, dy, dz, val);
                }
                loopStep = 1;
            }
            if (loopStep == 1) {
                if ((j & mask) < h) {
                    setInChunk(2, [c[0], c[1]-1, c[2]], i, j, k, dx, dy+chunkSize, dz, val);
                } else if ((j & mask) >= chunkSize - h) {
                    setInChunk(2, [c[0], c[1]+1, c[2]], i, j, k, dx, dy-chunkSize, dz, val);
                }
                loopStep = 2;
            }
            if (loopStep == 2) {
                if ((k & mask) < h) {
                    setInChunk(3, [c[0], c[1], c[2]-1], i, j, k, dx, dy, dz+chunkSize, val);
                } else if ((k & mask) >= chunkSize - h) {
                    setInChunk(3, [c[0], c[1], c[2]+1], i, j, k, dx, dy, dz-chunkSize, val);
                }
            }
        }

        var idx = 0;
        for (var i=ox; i < ox+region.dimensions[0]; i++) {
            for (var j=oy; j < oy+region.dimensions[1]; j++) {
                for (var k=oz; k < oz+region.dimensions[2]; k++, idx++) {
                    var val = this.chunkConverter.translateBlockID(region.voxelsU32.voxelsU32[idx]);
                    var c = game.voxels.chunkAtPosition([i,j,k]);
                    setInChunk(0, c, i, j, k, 0, 0, 0, val);
                }
            }
        }
    }
}

ChunkLoader.prototype.invalidateAllChunks = function(suppressReload) {
    var game = this.game;

    Object.keys(game.voxels.chunks).map(function(chunkIndex) {
        var chunk = game.voxels.chunks[chunkIndex]
        var mesh = game.voxels.meshes[chunkIndex]
        var pendingIndex = game.pendingChunks.indexOf(chunkIndex)
        if (pendingIndex !== -1) game.pendingChunks.splice(pendingIndex, 1)
        if (!chunk) return
        var chunkPosition = chunk.position
        if (mesh) {
          // dispose of the gl-vao meshes
          for (var key in mesh.vertexArrayObjects) {
            mesh.vertexArrayObjects[key].dispose()
          }
        }
        delete game.voxels.chunks[chunkIndex]
        delete game.voxels.meshes[chunkIndex]
        if (!suppressReload) {
            game.emit('removeChunk', chunkPosition)
        }
      })

    if (!suppressReload) {
        game.voxels.requestMissingChunks(game.playerPosition());
    }
}
