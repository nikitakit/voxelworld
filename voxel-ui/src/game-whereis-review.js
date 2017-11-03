"use strict";
var $ = require('jquery');
var ndarray = require('ndarray')
const infernoScale = require('scale-color-perceptual/inferno')
const hex2rgb = require('scale-color-perceptual/utils/hex2rgb')

function whereisReviewGame(tracker, dataContainer) {
    var game = window.game;
    var snapshot = dataContainer.snapshot;
    var response = dataContainer.data.response;
    var taskUtils = game.plugins.get('task-utils');

    var outlinePlugin = game.plugins.get('voxel-outline');
    var heatmapPlugin = game.plugins.get('heatmap');

    taskUtils.initTask();
    game.chunkLoader.invalidateAllChunks(true);
    taskUtils.addCleanup(function(){
        $("#task-container").empty();
        outlinePlugin.showOutline = false;
        outlinePlugin.showThrough = false;
        outlinePlugin.selectAdjacent = true;

        heatmapPlugin.setHeatmap(null);
        heatmapPlugin.showThrough = false;
    })

    outlinePlugin.showOutline = true;

    $("#task-container").empty();
    $("#task-container").append(""
        + "<p><button id='submit-review'>Request next task</button></p>"
        + "<p><button id='reset-camera-whereis'>Reset camera</button>"
        + "<button id='teleport-camera-whereis'>Zoom to Misty</button></p>"
        + "<p><button id='replay-whereis'>Replay movement</button> <button id='stop-replay-whereis'>Stop</button></p>"
        + "<input id='review-showthrough' type='checkbox'> Seethrough </input>"
        + "<input id='review-adjacent' type='checkbox' checked> Select adjacent </input>"
        + "<p>Heatmap: <select id='review-heatmaps'></select>"
        + "<button id='review-heatmap-render'>Render</button></p>"
        + "<p>Selected voxel: <span id='selected-voxel'></span></p>"
        + "<div style='position:absolute;bottom:1em'><h3>Annotation</h3>"
        + "<span id='annotation-review'></span>"
        + "</div>"
    );

    $('#review-showthrough').click(function() {
        outlinePlugin.showThrough = $("#review-showthrough").prop('checked');
        heatmapPlugin.showThrough = $("#review-showthrough").prop('checked');
    });

    $('#review-adjacent').click(function() {
        outlinePlugin.selectAdjacent = $("#review-adjacent").prop('checked');
    });

    taskUtils.restoreSnapshot(snapshot);
    $('#reset-camera-whereis').click(function()
    {
        taskUtils.restoreSnapshotCamera(snapshot);
    })
    $('#teleport-camera-whereis').click(function()
    {
        taskUtils.restoreSnapshotCamera({
            position: [dataContainer.data.x + 1, dataContainer.data.y + 1, dataContainer.data.z + 1],
            rotation: [Math.PI * 1 / 4, -Math.PI * 1 / 4, 0]
        });
    })

    var poseHistory = response.dataStruct.pose_history;
    var replayTimeout = null;
    function showKeyframe(i) {
        console.log("keyframe", i, poseHistory[i*7]);
        // sanity check only
        if ((i * 7 + 6) >= poseHistory.length) {
            return;
        }

        taskUtils.restoreSnapshotCamera({
            position: [poseHistory[i*7 + 1], poseHistory[i*7 + 2], poseHistory[i*7 + 3]],
            rotation: [poseHistory[i*7 + 4], poseHistory[i*7 + 5], poseHistory[i*7 + 6]]
        });

        if ((i+1) * 7 >= poseHistory.length) {
            console.log("done replaying movement");
            return;
        }

        var timeDelta = poseHistory[(i+1)*7] - poseHistory[i*7];
        console.log("keyframe", i, "holding for", timeDelta);
        if (timeDelta > 5) {
            timeDelta = 5;
        }

        replayTimeout = setTimeout(function() {
            showKeyframe(i+1);
        }, timeDelta*1000);
    }

    $('#replay-whereis').click(function(){
        poseHistory = response.dataStruct.pose_history;
        showKeyframe(0);
    })

    $('#stop-replay-whereis').click(function(){
        console.log('Stopping history replay');
        clearTimeout(replayTimeout);
        replayTimeout = null;
        poseHistory = [];
    })

    console.log('data struct', response.dataStruct);
    $('#annotation-review').html(response.dataStruct.annotation);

    game.setBlock([dataContainer.data.x, dataContainer.data.y, dataContainer.data.z],
        taskUtils.mistyId);
    game.updateDirtyChunks();



    var heatmaps = {};
    $("#review-heatmaps").empty();
    $("#review-heatmaps").append("<option>[none]</option>");
    $("#review-heatmaps").append("<option>candidates</option>");
    for (var i = 0; i < dataContainer.heatmaps.length; i++) {
        var optionElement = $("<option>" + dataContainer.heatmaps[i].name + "</option>");
        $("#review-heatmaps").append(optionElement);
        heatmaps[dataContainer.heatmaps[i].name] = dataContainer.heatmaps[i];
    }


    $('#review-heatmap-render').click(function(){
        var heatmapName = $('#review-heatmaps').find('option:selected').val();
        if (!heatmapName || heatmapName == "[none]") {
            heatmapPlugin.setHeatmap(null);
            return;
        }

        if (heatmapName == "candidates") {
            var region = snapshot.regions[0];
            var dims = [region.dimensions[0], region.dimensions[1], region.dimensions[2], 4];
            var heatmap = ndarray(new Float32Array(dims[0] * dims[1] * dims[2] * dims[3]), dims);
            var candidates = dataContainer.data.candidates;
            for (var i=0; i < candidates.length; i+=3) {
                var pos = [candidates[i] - region.position[0], candidates[i+1] - region.position[1], candidates[i+2] - region.position[2]];
                heatmap.set(pos[0], pos[1], pos[2], 0, 1);
                heatmap.set(pos[0], pos[1], pos[2], 3, 1);
            }
            heatmapPlugin.setHeatmap(region.position, heatmap);
            return;
        }

        var heatmap = heatmaps[heatmapName];
        if (!heatmap) {
            return;
        }

        // Minimum float value to display at all, instead of leaving the voxel
        // blank. By default, make it 1/10th of what the value would be if the
        // entire distribution were uniform
        var minToDisplay = 0.1 / (heatmap.dimensions[0] * heatmap.dimensions[1] * heatmap.dimensions[2]);

        var dims = [heatmap.dimensions[0], heatmap.dimensions[1], heatmap.dimensions[2], 4];
        var colorArray = ndarray(new Float32Array(dims[0] * dims[1] * dims[2] * dims[3]), dims);
        var heatmapIdx = 0;
        for (var i=0; i < dims[0]; i++) {
            for (var j=0; j < dims[1]; j++) {
                for (var k=0; k < dims[2]; k++, heatmapIdx++) {
                    if (heatmap.voxels[heatmapIdx] < minToDisplay) {
                        continue;
                    }
                    var colorHex = infernoScale(heatmap.voxels[heatmapIdx]);
                    var colorRGB = hex2rgb(colorHex);
                    colorArray.set(i, j, k, 0, colorRGB.r);
                    colorArray.set(i, j, k, 1, colorRGB.g);
                    colorArray.set(i, j, k, 2, colorRGB.b);
                    colorArray.set(i, j, k, 3, 1.);
                }
            }
        }
        heatmapPlugin.setHeatmap(heatmap.position, colorArray);
    })

    var displayBlockInfo = function() {
        if (!game.shell.pointerLock) {
            return;
        }

        var targetVoxel = game.plugins.get('voxel-outline').currentTarget;
        if (!targetVoxel) {
            return;
        }

        var voxelId = game.getBlock([targetVoxel[0], targetVoxel[1], targetVoxel[2]]);
        var minecraftId = game.chunkLoader.chunkConverter.reverseBlockIDs[voxelId];
        $('#selected-voxel').html('' + (minecraftId & 0xff) + ":" + (minecraftId >> 8));
    }

    game.on('tick', displayBlockInfo);
    taskUtils.addCleanup(function() {
        game.removeListener('tick', displayBlockInfo);
    });

    $('#submit-review').click(function(){
        tracker.submit_task({});
        // omit clearing, until next task comes in
    });
}

module.exports = function(tracker) {
    tracker.register_game("whereis_review_v1", function(data) {
        console.log('got whereis review game', data)
        whereisReviewGame(tracker, data.dataStaticWorld);
    });
}
