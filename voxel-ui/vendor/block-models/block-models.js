'use strict';

var vector = require('gl-vec3')

// get all coordinates for a cube ranging from vertex a to b
//   ____
//  /   /|
// a---+ |
// |   | b
// +---+/
var cubePositions = function(a, b) {
  return [
    [a[0], a[1], a[2]], // 0
    [a[0], a[1], b[2]], // 1
    [a[0], b[1], a[2]], // 2
    [a[0], b[1], b[2]], // 3
    [b[0], a[1], a[2]], // 4
    [b[0], a[1], b[2]], // 5
    [b[0], b[1], a[2]], // 6
    [b[0], b[1], b[2]], // 7
  ];
};

// get plane triangle vertices for given normal vector
function faceToMeshQuad(boxFrom, boxTo, direction) {
    var cubePos = cubePositions(boxFrom, boxTo);

    var mappings = {
        // TODO: these are probably wrong (especially up/down are suspect)
        // the interaction with uncompressUV needs to be considered here
        north: [cubePos[6], cubePos[2], cubePos[0], cubePos[4]],
        south: [cubePos[3], cubePos[7], cubePos[5], cubePos[1]],
        west: [cubePos[2], cubePos[3], cubePos[1], cubePos[0]],
        east: [cubePos[7], cubePos[6], cubePos[4], cubePos[5]],
        up: [cubePos[6], cubePos[7], cubePos[3], cubePos[2]],
        down: [cubePos[5], cubePos[4], cubePos[0], cubePos[1]],
    };

    return mappings[direction];
};

function uncompressUV(uv, rotation, output) {
    // Only support a rotation of 0 or 90 for now, because others have not
    // been observed in the data.
    // However, ideally 0, 90, 180, 270 will all be supported
    // Returns: 4 points TOP, RIGHT, BOTTOM, LEFT

    // TODO: this is probably wrong (at least the 90-degree rotation).
    // The rotations should probably preserve circular order, but they
    // currently don't
    if (rotation == 0) {
        output[0][0] = uv[0];
        output[0][1] = uv[1];
        output[1][0] = uv[2];
        output[1][1] = uv[1];
        output[2][0] = uv[2];
        output[2][1] = uv[3];
        output[3][0] = uv[0];
        output[3][1] = uv[3];
    } else {
        output[0][0] = uv[0];
        output[0][1] = uv[3];
        output[1][0] = uv[2];
        output[1][1] = uv[3];
        output[2][0] = uv[2];
        output[2][1] = uv[1];
        output[3][0] = uv[0];
        output[3][1] = uv[1];
    }
}

function rescaleUV(uv, origin, xDirection, yDirection) {
    for (var i = 0; i < 4; i++) {
        var xOffset = uv[i][0] / 16.0;
        var yOffset = uv[i][1] / 16.0;

        uv[i][0] = origin[0] + xOffset * xDirection[0] + yOffset * yDirection[0];
        uv[i][1] = origin[1] + xOffset * xDirection[1] + yOffset * yDirection[1];
    }
}

function appendQuad2D(list, quad) {
    // The order of points in the triangles matters, because the texture is
    // typically rendered when viewed from one (outward) direction, but not from
    // the opposite side.

    // The opposite-direction triangles are commented out here because they
    // are mostly useful for visual debugging of the block model renderer

    // list.push(quad[0][0]); list.push(quad[0][1]);
    // list.push(quad[2][0]); list.push(quad[2][1]);
    // list.push(quad[3][0]); list.push(quad[3][1]);
    //
    // list.push(quad[0][0]); list.push(quad[0][1]);
    // list.push(quad[1][0]); list.push(quad[1][1]);
    // list.push(quad[2][0]); list.push(quad[2][1]);

    list.push(quad[0][0]); list.push(quad[0][1]);
    list.push(quad[3][0]); list.push(quad[3][1]);
    list.push(quad[2][0]); list.push(quad[2][1]);

    list.push(quad[0][0]); list.push(quad[0][1]);
    list.push(quad[2][0]); list.push(quad[2][1]);
    list.push(quad[1][0]); list.push(quad[1][1]);
}

function appendQuad3D(list, quad) {
    // See comment in appendQuad2D above
    // list.push(quad[0][0]); list.push(quad[0][1]); list.push(quad[0][2]);
    // list.push(quad[2][0]); list.push(quad[2][1]); list.push(quad[2][2]);
    // list.push(quad[3][0]); list.push(quad[3][1]); list.push(quad[3][2]);
    //
    // list.push(quad[0][0]); list.push(quad[0][1]); list.push(quad[0][2]);
    // list.push(quad[1][0]); list.push(quad[1][1]); list.push(quad[1][2]);
    // list.push(quad[2][0]); list.push(quad[2][1]); list.push(quad[2][2]);

    list.push(quad[0][0]); list.push(quad[0][1]); list.push(quad[0][2]);
    list.push(quad[3][0]); list.push(quad[3][1]); list.push(quad[3][2]);
    list.push(quad[2][0]); list.push(quad[2][1]); list.push(quad[2][2]);

    list.push(quad[0][0]); list.push(quad[0][1]); list.push(quad[0][2]);
    list.push(quad[2][0]); list.push(quad[2][1]); list.push(quad[2][2]);
    list.push(quad[1][0]); list.push(quad[1][1]); list.push(quad[1][2]);
}

var _scratch_point = [0, 0, 0];
function rotatePoints3D(points, origin, axis, angle, rescale) {
    // points: array
    // origin: 3d point
    // axis: 'x', 'y', or 'z'
    // angle: -45, -22.5, 0, 22.5, or 45
    // rescale: bool
    if (angle == 0) {
        return;
    }

    angle = Math.PI * angle / 180.;
    // TODO: don't ignore rescale param

    for (var i = 0; i < points.length; i+=3) {
        _scratch_point[0] = points[i];
        _scratch_point[1] = points[i+1];
        _scratch_point[2] = points[i+2];

        // var point = [points[i], points[i+1], points[i+2]];
        if (axis == 'x') {
            vector.rotateX(_scratch_point, _scratch_point, origin, angle); // TODO: maybe negate angle?
        } else if (axis == 'y') {
            vector.rotateY(_scratch_point, _scratch_point, origin, angle); // TODO: maybe negate angle?
        } else if (axis == 'z') {
            vector.rotateZ(_scratch_point, _scratch_point, origin, angle); // angle is correct (theory+practice)
        }
        // points[i] = point[0];
        // points[i+1] = point[1];
        // points[i+2] = point[2];
        points[i] = _scratch_point[0];
        points[i+1] = _scratch_point[1];
        points[i+2] = _scratch_point[2];
    }
}

function rotatePoints3DAligned(points, x, y) {
    // x and y are both in multiples of 90 degrees, so it should be possible to
    // optimize this function instead of delegating to the general-purpose
    // rotation routines that use sin/cos
    if (x == 0 && y == 0) {
        return;
    }

    for (var i = 0; i < points.length; i+=3) {
        var point = [points[i], points[i+1], points[i+2]];
        if (x != 0) {
            // angle direction should be correct in theory
            vector.rotateX(point, point, [0.5, 0.5, 0.5], Math.PI * -x / 180.);
        }
        if (y != 0) {
            // angle direction is correct (theory+practice)
            vector.rotateY(point, point, [0.5, 0.5, 0.5], Math.PI * -y / 180.);
        }
        points[i] = point[0];
        points[i+1] = point[1];
        points[i+2] = point[2];
    }
}

function offsetPoints3D(points, x, y, z) {
    for (var i = 0; i < points.length; i+=3) {
        points[i] += x;
        points[i+1] += y;
        points[i+2] += z;
    }
}


// convert one JSON element to its vertices
var _scratch_faceUV = [[0,0], [0,0], [0,0], [0,0]];
var element2vertices = function(element, getTextureUV) {
    var boxFrom = [element.from[0] / 16.0, element.from[1] / 16.0, element.from[2] / 16.0];
    var boxTo =   [element.to  [0] / 16.0, element.to  [1] / 16.0, element.to  [2] / 16.0];

    var vertices = [];
    var uvArray = [];

    // add cells for each cube face (plane) in this element
    for (var direction in element.faces) {
      // position
      var faceInfo = element.faces[direction];
      var texture = faceInfo.texture || element.texture;

      // First determine the mesh quad
      var meshQuad = faceToMeshQuad(boxFrom, boxTo, direction);

      // Next determine the UV quad
      var textureUV = getTextureUV(texture); // array of 2d points [TOP, RIGHT, BOTTOM, LEFT]
      // We assume that the texture is a rectangle in UV space, which should be
      // true for our texture atlas.
      var uvOrigin = textureUV[0]; // TOP (0,0)
      var uvXDirection = [textureUV[1][0] - uvOrigin[0], textureUV[1][1] - uvOrigin[1]]; // RIGHT (1,0) - TOP (0,0)
      var uvYDirection = [textureUV[3][0] - uvOrigin[0], textureUV[3][1] - uvOrigin[1]]; // LEFT (0,1) - TOP (0,0)

      if (faceInfo.uv === undefined) {
          continue; //TODO(nikita): figure out how to fill this in
      }
      uncompressUV(faceInfo.uv, faceInfo.rotation || 0, _scratch_faceUV);
      rescaleUV(_scratch_faceUV, uvOrigin, uvXDirection, uvYDirection);

      appendQuad3D(vertices, meshQuad);
      appendQuad2D(uvArray, _scratch_faceUV);
    }

    if (element.rotation !== undefined && element.rotation.axis !== undefined) {
        var axis = element.rotation['axis'];
        var origin = element.rotation['origin'] || [8,8,8];
        origin = [origin[0] / 16., origin[1] / 16., origin[2] / 16.];
        var angle = element.rotation['angle'] || 0;
        var rescale = element.rotation['rescale'] || false;
        rotatePoints3D(vertices, origin, axis, angle, rescale);
    }

    rotatePoints3DAligned(vertices, element.xRot||0, element.yRot||0);

    return {vertices:vertices, uv:uvArray};
}

// convert an array of multiple cuboid elements
var elements2vertices = function(elements, getTextureUV, x, y, z) {
  var result = {vertices:[], uv:[]};

  for (var i = 0; i < elements.length; i += 1) {
    var element = elements[i];
    var thisResult = element2vertices(element, getTextureUV);

    result.vertices = result.vertices.concat(thisResult.vertices);
    result.uv = result.uv.concat(thisResult.uv);
  }

  offsetPoints3D(result.vertices, x, y, z);

  return result;
};

var parseBlockModel = function(elements, getTextureUV, x, y, z) {
  return elements2vertices(elements, getTextureUV, x|0, y|0, z|0);
};


module.exports = parseBlockModel;
