'use strict';

var createVoxelMesh = require('./mesh-buffer.js');
var inherits = require('inherits');
var EventEmitter = require('events').EventEmitter;
var ndarray = require('ndarray');
var ops = require('ndarray-ops');
var parseBlockModel = require("block-models");
var createBuffer = require("gl-buffer");
var createVAO = require("gl-vao");

module.exports = function(game, opts) {
  return new MesherPlugin(game, opts);
};
module.exports.pluginInfo = {
  loadAfter: ['voxel-registry', 'voxel-stitch'],
  clientOnly: true
};

function MesherPlugin(game, opts) {
  this.game = game;
  this.shell = game.shell;

  this.registry = game.plugins.get('voxel-registry');
  if (!this.registry) throw new Error('voxel-mesher requires voxel-registry plugin');

  this.stitcher = game.plugins.get('voxel-stitch');
  if (!this.stitcher) throw new Error('voxel-mesher requires voxel-stitch plugin');

  this.isTransparent = undefined;
  this.hasBlockModel = undefined;

  var s = game.chunkSize + (game.chunkPad|0)
  this.solidVoxels = ndarray(new game.arrayType(s*s*s), [s,s,s]);
};
inherits(MesherPlugin, EventEmitter);

MesherPlugin.prototype.createVoxelMesh = function(gl, voxels, voxelSideTextureIDs, voxelSideTextureSizes, position, pad) {
  // ADDED(nikita): caching uv coordinates gives major speed improvements, because
  // generating them requires a loop over the full atlas.
  // TODO(nikita): figure out if it's possible to cache even more aggresively
  this.uvCache = this.stitcher.atlas.uv(); // debugging note: array or not? https://github.com/shama/atlaspack/issues/5
  this.uvProcessed = {};

  var porousMesh = this.splitVoxelArray(voxels);

  var mesh = createVoxelMesh(gl, this.solidVoxels, voxelSideTextureIDs, voxelSideTextureSizes, position, pad, this);

  mesh.vertexArrayObjects.porous = porousMesh;

  return mesh;
}


// ADDED(nikita)
// This is a hacked version based on https://github.com/voxel/voxel-stitch/blob/master/stitch.js
// The call to uv() is expensive, and perf showed that it was taking up the
// majority of CPU time for a loading a world chunk-by-chunk.
function getTextureUVCached(mesher, stitcher, name) {
    var uv = mesher.uvCache[name];
    if (!uv) return undefined;

    if (mesher.uvProcessed[name]) {
        return uv;
    }

    uv = uv.slice();

    var d = stitcher.tileSize / stitcher.atlasSize;

    // unpad from the 2x2 repeated tiles, so we only return one
    uv[1][0] -= d * (stitcher.tilePad - 1);
    uv[2][0] -= d * (stitcher.tilePad - 1);
    uv[2][1] -= d * (stitcher.tilePad - 1);
    uv[3][1] -= d * (stitcher.tilePad - 1);

    mesher.uvProcessed[name] = true;
    return uv;
}

// mesh custom voxel
MesherPlugin.prototype.meshCustomBlock = function(value,x,y,z) {
  var modelDefn = this.registry.blockProps[value].blockModel;
  var stitcher = this.stitcher;
  var self = this;

  // parse JSON to vertices and UV
  var model = parseBlockModel(
    modelDefn,
    //getTextureUV:
    function(name) {
        return getTextureUVCached(self, stitcher, name);
      //return stitcher.getTextureUV(name); // only available when textures are ready
    },
    x,y,z
  );

  return model;
};

// PERF(nikita): profiling chunk loading/rendering seems to show that the
// majority of the time is spent inside this method (self time).
// It's worth looking at whether there's any way to speed up the code here.
// Time attributed to function calls out of here is much smaller than the
// self time
//
// populates solidVoxels array, returns porousMesh
MesherPlugin.prototype.splitVoxelArray = function(voxels) {
  if (!this.isTransparent) {
    // cache list of transparent voxels TODO: refresh cache when changes
    this.isTransparent = this.registry.getBlockPropsAll('transparent');
    this.isTransparent.unshift(true); // air (0) is transparent
  }
  if (!this.hasBlockModel) {
    this.hasBlockModel = this.registry.getBlockPropsAll('blockModel');
    this.hasBlockModel.unshift(undefined);
  }

  // phase 1: solid voxels = opaque, transparent (terrain blocks, glass, greedily meshed)
  var solidVoxels = this.solidVoxels;
  var isTransparent = this.isTransparent;
  ops.assign(solidVoxels, voxels);

  // phase 2: porous voxels = translucent, custom block models (stained glass, slabs, stairs)
  var hasBlockModel = this.hasBlockModel;
  var porousMeshes = this.porousMeshes = [];

  var length = solidVoxels.data.length;
  var vertices = [];
  var uv = [];
  for (var i = 0; i < length; ++i) {
    var value = solidVoxels.data[i];
    if (hasBlockModel[value]) {
      solidVoxels.data[i] = 0;

      // HACK: the code here assumes chunk size of 32 with 2 padding on
      // each side. This is left-over from the original author.
      // In the future, it would be good to switch
      // to a more portable index extraction scheme.
      var o = i;
      var z = (o % 36)-2; o = Math.floor(o / 36);
      var y = (o % 36)-2; o = Math.floor(o / 36);
      var x = o-2;

      // accumulate mesh vertices and uv
      var model = this.meshCustomBlock(value,x,y,z);
      for (var j = 0; j < model.vertices.length; j++) {
          vertices.push(model.vertices[j]);
      }
      for (var k = 0; k < model.uv.length; k++) {
          uv.push(model.uv[k]);
      }
    } else if (!isTransparent[value]) {
      solidVoxels.data[i] = value | (1<<15); // opaque bit
    }
  }

  // load combined porous mesh into GL
  var gl = this.shell.gl;
  var verticesBuf = createBuffer(gl, new Float32Array(vertices));
  var uvBuf = createBuffer(gl, new Float32Array(uv));
  var porousMesh = createVAO(gl, [
      { buffer: verticesBuf,
        size: 3
      },
      {
        buffer: uvBuf,
        size: 2
      }
      ]);
  porousMesh.length = vertices.length/3;

  return porousMesh;
};
