'use strict';

// TODO: the entire renderer/shader implementation here is a mess, and should
// probably be rewritten

var createBuffer = require('gl-buffer');
var createVAO = require('gl-vao');
var createShader = require("gl-shader")
var mat4 = require('gl-mat4');
var inherits = require('inherits');
var EventEmitter = require('events').EventEmitter;

var ndarray = require('ndarray')

module.exports = function(game, opts) {
  return new HeatmapPlugin(game, opts);
};
module.exports.pluginInfo = {
  loadAfter: ['voxel-mesher', 'voxel-shader']
};

function createCustomShader(gl) {
return createShader(gl,
"attribute vec3 position;\
attribute vec3 color;\
uniform mat4 model;\
uniform mat4 view;\
uniform mat4 projection;\
varying vec3 fragColor;\
void main() {\
  gl_Position = projection * view * model * vec4(position,1);\
  fragColor = color;\
}",
"precision highp float;\
varying vec3 fragColor;\
void main() {\
  gl_FragColor = vec4(fragColor, 0.3);\
}")
}

function HeatmapPlugin(game, opts) {
  this.game = game;
  this.shell = game.shell;

  this.mesherPlugin = game.plugins.get('voxel-mesher');
  if (!this.mesherPlugin) throw new Error('heat-map-3d requires voxel-mesher');

  this.shaderPlugin = game.plugins.get('voxel-shader');
  if (!this.shaderPlugin) throw new Error('heat-map-3d requires voxel-shader');

  this.showThrough = opts.showThrough !== undefined ? opts.showThrough : false;

  this.heatmap = null;
  this.scratchMatrix = mat4.create();

  this.enable();
}
inherits(HeatmapPlugin, EventEmitter);

HeatmapPlugin.prototype.enable = function() {
  this.shell.on('gl-init', this.onInit = this.shaderInit.bind(this));
  this.shell.on('gl-render', this.onRender = this.render.bind(this));
};

HeatmapPlugin.prototype.disable = function() {
  this.shell.removeListener('gl-render', this.onRender = this.render.bind(this));
  this.shell.removeListener('gl-init', this.onInit);
  this.currentTarget = undefined;
};

var originOffsetVector = [0.5 - 0.15, 0.5 - 0.15, 0.5 - 0.15];
var scaleVector = [0.3, 0.3, 0.3];

HeatmapPlugin.prototype.setHeatmap = function(origin, heatmap) {
    if (!origin || !heatmap) {
        this.heatmap = null;
        return;
    }

    this.heatmap = heatmap;
    this.origin = origin;
};

HeatmapPlugin.prototype.test = function() {
    var origin = [0,0,0];
    var dims = [2,2,2,4];
    var heatmap = ndarray(new Uint16Array(dims[0] * dims[1] * dims[2] * dims[3]), dims);
    heatmap.set(0,0,0,3, 1);
    heatmap.set(1,1,1,3, 1);
    heatmap.set(2,2,2,3, 1);
    this.setHeatmap(origin, heatmap);
}

HeatmapPlugin.prototype.render = function() {
    if (!this.origin || !this.heatmap) {
        return;
    }

    var gl = this.shell.gl;

    if (this.showThrough) {
        gl.disable(gl.DEPTH_TEST);
        gl.enable(gl.BLEND);
    }

    this.heatmapShader.attributes.position.location = 0;
    this.heatmapShader.attributes.color.location = 1;
    this.heatmapShader.bind();
    this.heatmapShader.uniforms.projection = this.shaderPlugin.projectionMatrix;
    this.heatmapShader.uniforms.view = this.shaderPlugin.viewMatrix;

    var indicatorVAO = this.mesh;

    for (var i = 0; i < this.heatmap.shape[0]; i++) {
        for (var j = 0; j < this.heatmap.shape[1]; j++) {
            for (var k = 0; k < this.heatmap.shape[2]; k++) {
                // TODO(nikita): the intent is to have an alpha channel, but
                // right now we just treat it as a boolean mask
                if (this.heatmap.get(i,j,k,3) < 0.001) {
                    continue;
                }
                var color = [this.heatmap.get(i,j,k,0), this.heatmap.get(i,j,k,1), this.heatmap.get(i,j,k,2)];

                mat4.identity(this.scratchMatrix);
                mat4.translate(this.scratchMatrix, this.scratchMatrix, this.origin);
                mat4.translate(this.scratchMatrix, this.scratchMatrix, originOffsetVector);
                mat4.translate(this.scratchMatrix, this.scratchMatrix, [i, j, k]);
                mat4.scale(this.scratchMatrix, this.scratchMatrix, scaleVector);
                this.heatmapShader.attributes.color = color;
                this.heatmapShader.uniforms.model = this.scratchMatrix;
                indicatorVAO.bind();
                indicatorVAO.draw(gl.TRIANGLES, indicatorVAO.length);
                indicatorVAO.unbind();
            }
        }
    }
};

HeatmapPlugin.prototype.shaderInit = function() {
  this.heatmapShader = createCustomShader(this.shell.gl);

  var w = 1;
  var outlineVertexArray = new Uint8Array([
    0,0,0,
    0,0,w,
    0,w,0,
    0,w,w,
    w,0,0,
    w,0,w,
    w,w,0,
    w,w,w
  ]);

  var indexArray = new Uint16Array([
      0,1,2, 1,3,2,
      5,1,4, 1,0,4,
      6,2,7, 2,3,7,
      4,0,6, 0,2,6,
      7,3,1, 7,1,5,
      4,6,5, 5,6,7,
  ]);

  var vertexCount = indexArray.length;

  var gl = this.shell.gl;

  var outlineBuf = createBuffer(gl, outlineVertexArray);
  var indexBuf = createBuffer(gl, indexArray, gl.ELEMENT_ARRAY_BUFFER);

  var indicatorVAO = createVAO(gl, [
      { buffer: outlineBuf,
        type: gl.UNSIGNED_BYTE,
        size: 3
      }], indexBuf);
  indicatorVAO.length = vertexCount;

  this.mesh = indicatorVAO;
};
