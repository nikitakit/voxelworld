"use strict";

module.exports = function(game, opts) {
    return new TaskUtils(game);
}

module.exports.pluginInfo = {
    loadAfter: ['voxel-artpacks', 'voxel-stitch', 'chunk-loader']
}

const MISTY_BLOCK = {
    blockModel: [{
        from: [5,0,5],
        to: [10,5,10],
        faces: {
            north: {
                "uv": [5,0,10,5],
                "texture": "misty"
            },
            south: {
                "uv": [5,5,10,10],
                "texture": "misty"
            },
            east: {
                "uv": [10,5,15,10],
                "texture": "misty"
            },
            west: {
                "uv": [0,5,5,10],
                "texture": "misty"
            },
            up: {
                "uv": [5,0,10,5],
                "texture": "misty"
            },
            down: {
                "uv": [5,0,10,5],
                "texture": "misty"
            },
        },
        yRot: 45
    }]
};

const MISTY_MISS_BLOCK = {
    blockModel: [{
        from: [7,0,7],
        to: [8,1,8],
        faces: {
            north: {
                "uv": [5,0,6,1],
                "texture": "misty"
            },
            south: {
                "uv": [5,0,6,1],
                "texture": "misty"
            },
            east: {
                "uv": [5,0,6,1],
                "texture": "misty"
            },
            west: {
                "uv": [5,0,6,1],
                "texture": "misty"
            },
            up: {
                "uv": [5,0,6,1],
                "texture": "misty"
            },
            down: {
                "uv": [5,0,6,1],
                "texture": "misty"
            },
        },
        yRot: 45
    }]
};

function makeLetterBlock(texture){
    return {
        blockModel: [{
            from: [5,0,5],
            to: [10,5,10],
            faces: {
                north: {
                    "uv": [0,0,9,9],
                    "texture": texture
                },
                south: {
                    "uv": [0,0,9,9],
                    "texture": texture
                },
                east: {
                    "uv": [0,0,9,9],
                    "texture": texture
                },
                west: {
                    "uv": [0,0,9,9],
                    "texture": texture
                },
                up: {
                    "uv": [0,0,9,9],
                    "texture": texture
                },
                down: {
                    "uv": [0,0,9,9],
                    "texture": texture
                },
            },
            yRot: 45
        }]
    };
};

function lockChangeAlert() {
  if(document.pointerLockElement !== null ||
      document.mozPointerLockElement !== null) {
      document.activeElement.blur(); // Unfocus to get controls back
  }
}

function TaskUtils(game, opts) {
    console.log("Loading custom blocks plugin");
    this.game = game;
    this.devMode = false;

    game.plugins.get('voxel-stitch').preloadTexture('misty');
    game.plugins.get('voxel-stitch').preloadTexture('letter_a');
    game.plugins.get('voxel-stitch').preloadTexture('letter_b');
    game.plugins.get('voxel-stitch').preloadTexture('letter_c');
    game.plugins.get('voxel-stitch').preloadTexture('letter_d');
    game.plugins.get('voxel-stitch').preloadTexture('letter_e');
    game.plugins.get('voxel-stitch').preloadTexture('letter_f');

    var registry = game.plugins.get('voxel-registry');
    registry.registerBlock('misty', MISTY_BLOCK);
    registry.registerBlock('misty-miss', MISTY_MISS_BLOCK);
    registry.registerBlock('letter-a', makeLetterBlock('letter_a'));
    registry.registerBlock('letter-b', makeLetterBlock('letter_b'));
    registry.registerBlock('letter-c', makeLetterBlock('letter_c'));
    registry.registerBlock('letter-d', makeLetterBlock('letter_d'));
    registry.registerBlock('letter-e', makeLetterBlock('letter_e'));
    registry.registerBlock('letter-f', makeLetterBlock('letter_f'));

    this.mistyId = registry.getBlockIndex('misty');
    this.missId = registry.getBlockIndex('misty-miss');

    this.cleanupDisposables = [];


    if ("onpointerlockchange" in document) {
      document.addEventListener('pointerlockchange', lockChangeAlert, false);
    } else if ("onmozpointerlockchange" in document) {
      document.addEventListener('mozpointerlockchange', lockChangeAlert, false);
    }
}

TaskUtils.prototype.setDev = function(devMode) {
    this.devMode = devMode;
}

TaskUtils.prototype.getLetterId = function(letter) {
    return this.game.plugins.get('voxel-registry').getBlockIndex('letter-'+letter);
}

TaskUtils.prototype.restoreSnapshot = function(snapshot) {
    if (!this.devMode) {
        this.game.chunkLoader.invalidateAllChunks(true);
    }

    self.game.controls.target().velocity = [0, 0, 0];

    var camera = this.game.cameraPlugin.camera;

    camera.position[0] = -snapshot.position[0];
    camera.position[1] = -snapshot.position[1];
    camera.position[2] = -snapshot.position[2];

    // rotation may not be right, check if any need to be negated
    camera.rotationX = snapshot.rotation[0];
    camera.rotationY = snapshot.rotation[1];
    camera.rotationZ = snapshot.rotation[2];

    if (snapshot.regions !== undefined) {
        for (var i = 0; i < snapshot.regions.length; i++) {
            this.game.chunkLoader.drawRegion(snapshot.regions[i]);
        }
    }
}

TaskUtils.prototype.restoreSnapshotCamera = function(snapshot) {
    self.game.controls.target().velocity = [0, 0, 0];

    var camera = this.game.cameraPlugin.camera;

    camera.position[0] = -snapshot.position[0];
    camera.position[1] = -snapshot.position[1];
    camera.position[2] = -snapshot.position[2];

    // rotation may not be right, check if any need to be negated
    camera.rotationX = snapshot.rotation[0];
    camera.rotationY = snapshot.rotation[1];
    camera.rotationZ = snapshot.rotation[2];
}

TaskUtils.prototype.initTask = function() {
    this.cleanupTask();
}

TaskUtils.prototype.cleanupTask = function () {
    for (var i = 0; i < this.cleanupDisposables.length; i++) {
        this.cleanupDisposables[i].dispose();
    }
}

TaskUtils.prototype.addCleanup = function (obj) {
    if (obj.dispose) {
        this.cleanupDisposables.push(obj);
    } else {
        this.cleanupDisposables.push(new CleanupDisposable(obj));
    }
}

TaskUtils.prototype.disableInteraction = function() {
    var self = this;
    var oldTick = self.game.controls.tick;

    if (!self.devMode) {
        // Disable moving around for non-interactive tasks
        self.game.controls.target().velocity = [0, 0, 0];
        self.game.controls.tick = function() {};

        // Don't request pointer lock for non-interactive tasks
        self.game.shell.pointerLock = false;
    }

    this.cleanupDisposables.push(new CleanupDisposable(
        function () {
            self.game.controls.tick = oldTick;
        }
    ))
}

TaskUtils.prototype.enableInteraction = function() {
    var self = this;

    if (!self.devMode) {
        self.game.shell.pointerLock = true;
    }
}

TaskUtils.prototype.monitorCameraMovement = function() {
    var self = this;

    // Course-grained trajectory monitoring
    var startTime = null;
    var data = {
        poseHistory: [],
        positionIntegral: 0
    };

    var lastKeyframePose = [Infinity, Infinity, Infinity, Infinity, Infinity, Infinity];
    var lastTickPose = [Infinity, Infinity, Infinity, Infinity, Infinity, Infinity];
    var newPose = [0, 0, 0, 0, 0, 0];

    var checkPositionChange = function() {
        if (startTime === null) {
            startTime = Date.now();
        }

        var newPosition = self.game.cameraPosition();
        newPose[0] = newPosition[0];
        newPose[1] = newPosition[1];
        newPose[2] = newPosition[2];
        newPose[3] = self.game.cameraPlugin.camera.rotationX;
        newPose[4] = self.game.cameraPlugin.camera.rotationY;
        newPose[5] = self.game.cameraPlugin.camera.rotationZ;

        if (lastTickPose[0] != Infinity) {
            // Update position integral, except on first timestep
            var distance2 = 0;
            for (var i = 0; i < 3; i++) {
                distance2 += Math.pow(newPose[i] - lastTickPose[i], 2);
            }
            data.positionIntegral += Math.sqrt(distance2);
        }

        // Given significant enough motion, update the poseHistory
        var diff = 0;
        for(var i = 0; i < 3; i++) {
            diff += Math.pow(newPose[i] - lastKeyframePose[i], 2);
        }

        for(var i = 3; i < 6; i++) {
            diff += 25. * Math.pow(newPose[i] - lastKeyframePose[i], 2);
        }

        if (diff > 1.) {
            data.poseHistory.push((Date.now() - startTime) / 1000);
            for(var i = 0; i < 6; i++) {
                data.poseHistory.push(newPose[i]);
                lastKeyframePose[i] = newPose[i];
            }
        }

        var tmp = lastTickPose;
        lastTickPose = newPose;
        newPose = tmp;
    }

    this.game.on('tick', checkPositionChange);
    this.addCleanup(function() {
        game.removeListener('tick', checkPositionChange);
    });
    return data;
}

function CleanupDisposable(cleanup) {
    this._cleanup = cleanup;
    this.disposed = false;
}

CleanupDisposable.prototype.dispose = function() {
    if (this.disposed) {
        return;
    }
    this.disposed = true;
    this._cleanup();
}
