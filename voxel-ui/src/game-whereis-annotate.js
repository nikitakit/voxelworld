"use strict";
var $ = require('jquery');
var validateText = require('./validate-text');

function whereisAnnotationGame(tracker, dataContainer, invalidate) {
    var game = window.game;
    var snapshot = dataContainer.snapshot;
    var taskUtils = game.plugins.get('task-utils');

    taskUtils.initTask();
    taskUtils.addCleanup(function(){
        $("#task-container").empty();
    })

    $("#task-container").empty();
    $("#task-container").append(""
        + "<img src='static/misty-portrait.png' style='width:50px;float:left;margin:2px'></img>"
        + "<p>Meet Misty.</p>"
        + "<p>Misty should be somewhere in the scene to the left. Please take a closer look at the scene by moving the camera around. Then use a few sentences to describe Misty's location.</p>"
        + "<p>(Misty loves to hide by going invisible. Your description should be specific enough to help someone locate Misty if she were to turn invisible.)</p>"
        + "<p><textarea id='annotation' style='width:90%;height:5em;'/><br/>"
        + "<button id='task-submit-button'>Submit</button></p>"
        + "<h3>How to move around</h3>"
        + "Click on the scene to the left to hide the mouse cursor."
        + " Then use the mouse and arrow keys to get a better look at the scene."
        + " Press ESC to unhide the mouse cursor. <br/>"
        + "<button id='reset-camera-whereis'>Click here to reset camera position</button>"
        + "<button id='teleport-camera-whereis'>Click here to zoom in on Misty</button>"
        + "<div style='position:absolute;bottom:1em'><h3>Controls</h3>"
        + "<p>WASD / arrow keys: move the camera<br/>"
        + "space: move up<br/>"
        + "shift: move down<br/>"
        + "esc: unhide the mouse cursor<br/>"
        + "</p>"
        + "</div>"
    );

    var button = $('#task-submit-button');

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

    if (taskUtils.devMode) {
        // Misty will overwrite the corresponding candidate block, so do this first
        var candidates = dataContainer.data.candidates;
        for (var i=0; i < candidates.length; i+=3) {
            game.setBlock([candidates[i], candidates[i+1], candidates[i+2]], taskUtils.missId);
        }
    }

    game.setBlock([dataContainer.data.x, dataContainer.data.y, dataContainer.data.z],
        taskUtils.mistyId);

    button.click(function() {
        if (movementData.positionIntegral < 1.) {
            alert("Before submitting, you must use the arrow keys to move the camera.\n\nFirst click into the scene on the left, then use the arrow keys to move.");
            return;
        }

        var text = $("#annotation").val();

        console.log("submitting annotation", text);

        var alertMsg = validateText(text);
        if (alertMsg !== null) {
            alert(alertMsg);
            return;
        }

        var result_data = {
            dataStruct: {
                annotation: text,
                pose_history: movementData.poseHistory,
                position_integral: movementData.positionIntegral
            }
        }

        tracker.submit_task(result_data);
        taskUtils.cleanupTask();
    });

    game.updateDirtyChunks();
    var movementData = taskUtils.monitorCameraMovement();
}

// Used during development, as part of creating the initial set of tasks
function whereisAnnotationCreateGame(tracker, dataContainer, invalidate) {
    var game = window.game;
    var snapshot = dataContainer.snapshot;
    var taskUtils = game.plugins.get('task-utils');
    game.chunkLoader.invalidateAllChunks(true);
    whereisAnnotationGame(tracker, dataContainer, invalidate);
    $("#task-container").empty();
    $("#task-container").append(""
        + "<p><button id='task-create-save'>Save task</button> </p>"
        + "<p><button id='task-create-reroll'>Reroll Misty's location</button> </p>"
        + "<p><button id='task-create-skip'>Skip this scene</button> </p>"
        + "<p></p>"
        + "<p><button id='reset-camera-create'>Click here to reset camera position</button></p>"
    );

    $('#reset-camera-create').click(function()
    {
        taskUtils.restoreSnapshotCamera(snapshot);
    })

    function submitTask(choice) {
        var result_data = {
            dataStruct: {
                choice: choice
            }
        }

        tracker.submit_task(result_data);
        taskUtils.cleanupTask();
    }

    $('#task-create-save').click(function(){
        submitTask('save');
    });
    $('#task-create-reroll').click(function(){
        submitTask('reroll');
    });
    $('#task-create-skip').click(function(){
        submitTask('skip');
    });
}

module.exports = function(tracker, invalidate) {
    tracker.register_game("whereis_annotate_v1", function(data) {
        console.log('got whereis game', data)
        whereisAnnotationGame(tracker, data.dataStaticWorld, invalidate);
    });
    tracker.register_game("whereis_annotate_create_v1", function(data) {
        whereisAnnotationCreateGame(tracker, data.dataStaticWorld, invalidate);
    });
}
