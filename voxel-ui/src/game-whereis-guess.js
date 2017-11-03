"use strict";
var $ = require('jquery');

function whereisGuessGameV1(tracker, dataContainer, invalidate) {
    // This is whereis_guess_v1, which has the user interactively
    // place Misty. This game is actually very hard, so starting with v2
    // we switched to a choice-out-of-5 approach.
    var game = window.game;
    var snapshot = dataContainer.snapshot;
    var taskUtils = game.plugins.get('task-utils');

    taskUtils.initTask();
    taskUtils.enableInteraction();

    var attempts = $('#attempts_left');
    var maxAttempts = 5;
    var attemptsNum = [-1];
    var attemptLocations = [];

    $("#task-container").empty();
    $("#task-container").append(""
        + "<img src='static/misty-portrait.png' style='width:50px;float:left;margin:2px'></img>"
        + "<p>Meet Misty.</p>"
        + "<p>Misty is invisible, hiding in an empty block in the scene to the left. You have been given a clue about her location:</p>"
        + "<p style='font-weight:bold;padding:4px;margin-right:6px;border-style:solid;border-color:black;border-width:2px;height:6em;overflow-y:scroll'>" + dataContainer.data.annotation + "</p>"
        + "<p>Your goal is to guess Misty's location.</p>"
        + "<p>Click inside the scene on the left to activate the game and hide your mouse pointer.</p>"
        + "<p>Then use the mouse and arrow keys to highlight the block you think Misty is in. (A red border indicates that a block is highlighted.) Click the mouse to make a guess.</p>"
        + "<p><strong>Attempts left: <a id='attempts-left'></a></strong></p>"
        + "<p><a style='color:red' id='try-again-text'></a></p>"
        + "<div style='position:absolute;bottom:1em'><h3>Controls</h3>"
        + "<p>WASD / arrow keys: move the camera<br/>"
        + "space: move up<br/>"
        + "shift: move down<br/>"
        + "click: make a guess<br/>"
        + "esc: unhide the mouse cursor<br/>"
        + "<button id='reset-camera'>Click here to reset camera position</button>"
        + "</p>"
        + "</div>"
    );
    var attempts = $('#attempts-left');
    var tryAgainText = $('#try-again-text');
    var tryAgainPrompts = [
        "Misty is not there! Please make another guess.",
        "Please try again! Misty is in a different block."
    ];

    function incrementAttempts() {
        attemptsNum[0] += 1;
        attempts.html("" + maxAttempts - attemptsNum[0]);
    }

    incrementAttempts();

    taskUtils.restoreSnapshot(snapshot);
    $('#reset-camera').click(function()
    {
        taskUtils.restoreSnapshotCamera(snapshot);
    })

    function clickListener(controlling, state) {
        var targetVoxel = game.plugins.get('voxel-outline').currentTarget;
        if (!targetVoxel) {
            return;
        }
        attemptLocations.push([targetVoxel[0], targetVoxel[1], targetVoxel[2]]);
        incrementAttempts();

        if (targetVoxel[0] == dataContainer.data.x
            && targetVoxel[1] == dataContainer.data.y
            && targetVoxel[2] == dataContainer.data.z) {
            game.setBlock(targetVoxel, taskUtils.mistyId);
            tryAgainText.html("");

            // Delay so misty is rendered before the alert
            setTimeout(function(){
                alert("You found Misty!");

                var result_data = {
                        dataStruct: {
                            guesses: attemptLocations,
                            found: true,
                            pose_history: movementData.poseHistory,
                            position_integral: movementData.positionIntegral
                        }
                    }
                tracker.submit_task(result_data);
                taskUtils.cleanupTask();
            }, 10);
            return;
        } else if (attemptsNum[0] >= maxAttempts) {
            game.setBlock(targetVoxel, taskUtils.missId);
            tryAgainText.html("Sorry, you've used all your attempts");

            alert("You failed to find Misty.");
            var result_data = {
                    dataStruct: {
                        guesses: attemptLocations,
                        found: false,
                        pose_history: movementData.poseHistory,
                        position_integral: movementData.positionIntegral
                    }
                }
            tracker.submit_task(result_data);
            taskUtils.cleanupTask();
        } else {
            game.setBlock(targetVoxel, taskUtils.missId);
            tryAgainText.html(tryAgainPrompts[Math.floor(Math.random()*tryAgainPrompts.length)])
        }
    }

    game.on('fire', clickListener);
    game.plugins.get('voxel-outline').showOutline = true;


    taskUtils.addCleanup(function(){
        $("#task-container").empty();
        game.removeListener('fire', clickListener);
        game.plugins.get('voxel-outline').showOutline = false;
    })

    game.updateDirtyChunks();
    var movementData = taskUtils.monitorCameraMovement();
}


function whereisGuessGameV2(tracker, dataContainer, invalidate) {
    var game = window.game;
    var snapshot = dataContainer.snapshot;
    var taskUtils = game.plugins.get('task-utils');

    taskUtils.initTask();

    var upToF = (dataContainer.data.choice_candidates.length > 15);

    $("#task-container").empty();
    $("#task-container").append(""
        + "<img src='static/misty-portrait.png' style='width:50px;float:left;margin:2px'></img>"
        + "<p>Meet Misty.</p>"
        + "<p>Misty is hidden somewhere in the scene to the left. You have been given a clue about her location:</p>"
        + "<p style='font-weight:bold;padding:4px;margin-right:6px;border-style:solid;border-color:black;border-width:2px;height:6em;overflow-y:scroll'>" + dataContainer.data.annotation + "</p>"
        + "<p>Please take a closer look at the scene on the left by clicking inside it, and then using the mouse and arrow keys to move around. Once you are done looking, press ESC to unhide the mouse cursor.</p>"
        + ((!upToF)
            ? "<p>Which of the five blocks marked A-E is most likely to contain Misty?</p>"
            : "<p>Which of the six blocks marked A-F is most likely to contain Misty?</p>")
        + "<input type='radio' name='task-location' value='0'>A &nbsp;&nbsp;&nbsp;"
        + "<input type='radio' name='task-location' value='1'>B &nbsp;&nbsp;&nbsp;"
        + "<input type='radio' name='task-location' value='2'>C &nbsp;&nbsp;&nbsp;"
        + "<input type='radio' name='task-location' value='3'>D &nbsp;&nbsp;&nbsp;"
        + ((!upToF)
            ? "<input type='radio' name='task-location' value='4'>E <br/>"
            : ("<input type='radio' name='task-location' value='4'>E &nbsp;&nbsp;&nbsp;"
               + "<input type='radio' name='task-location' value='5'>F <br/>"))
        + "<input type='checkbox' id='task-report'>Report clue as off-topic<br/>"
        + "<p><button id='task-submit-button'>Submit</button></p>"
        + "<div style='position:absolute;bottom:1em'><h3>Controls</h3>"
        + "<p>WASD / arrow keys: move the camera<br/>"
        + "space: move up<br/>"
        + "shift: move down<br/>"
        + "esc: unhide the mouse cursor<br/>"
        + "<button id='reset-camera'>Click here to reset camera position</button>"
        + "</p>"
        + "</div>"
    );

    taskUtils.restoreSnapshot(snapshot);
    $('#reset-camera').click(function()
    {
        taskUtils.restoreSnapshotCamera(snapshot);
    })

    var candidates = dataContainer.data.choice_candidates;
    var letters = ['a', 'b', 'c', 'd', 'e', 'f'];
    for (var i=0; i < candidates.length; i+=3) {
        game.setBlock([candidates[i], candidates[i+1], candidates[i+2]], taskUtils.getLetterId(letters[i/3]));
    }

    $('#task-submit-button').click(function() {
        if (movementData.positionIntegral < 1.) {
            alert("Before submitting, you must use the arrow keys to move the camera.\n\nFirst click into the scene on the left, then use the arrow keys to move.");
            return;
        }

        var choiceVal = $('input[name="task-location"]:checked').val()
        var reportVal = $('#task-report').prop('checked')

        if (choiceVal === undefined) {
            if (reportVal) {
                alert("Please select your best guess, even if the clue is not very useful.");

            } else if (!upToF) {
                alert("Please select an option A-E.\n\nIf the clue text is off-topic, you can check the box to report it.");
            } else {
                alert("Please select an option A-F.\n\nIf the clue text is off-topic, you can check the box to report it.");
            }
            return;
        }

        console.log("submitting choice", choiceVal);

        var result_data = {
            dataStruct: {
                choice: choiceVal,
                report: reportVal,
                pose_history: movementData.poseHistory,
                position_integral: movementData.positionIntegral
            }
        }

        $("#task-container").empty();

        if (dataContainer.data.x == candidates[choiceVal*3] &&
            dataContainer.data.y == candidates[choiceVal*3+1] &&
            dataContainer.data.z == candidates[choiceVal*3+2]) {
            $("#task-container").append("<p style='color:green'>Correct!</p>");
        } else {
            $("#task-container").append("<p style='color:red'>Sorry! You did not guess correctly.</p>");
        }

        for (var i=0; i < candidates.length; i+=3) {
            if (i == choiceVal * 3) {
                continue;
            }
            game.setBlock([candidates[i], candidates[i+1], candidates[i+2]], 0);
        }
        game.setBlock([dataContainer.data.x, dataContainer.data.y, dataContainer.data.z], taskUtils.mistyId);
        game.updateDirtyChunks();

        setTimeout(function() {
            tracker.submit_task(result_data);
            taskUtils.cleanupTask();
        }, 10);
    });

    game.updateDirtyChunks();
    var movementData = taskUtils.monitorCameraMovement();
}


module.exports = function(tracker, invalidate) {
    tracker.register_game("whereis_guess_v1", function(data) {
        console.log('got whereis guess game', data)
        whereisGuessGameV1(tracker, data.dataStaticWorld, invalidate);
    });
    tracker.register_game("whereis_guess_v2", function(data) {
        console.log('got whereis guess game v2', data)
        whereisGuessGameV2(tracker, data.dataStaticWorld, invalidate);
    });
}
