var $ = require('jquery');

function annotationGame(tracker, snapshot, invalidate) {
    var game = window.game;
    var taskUtils = game.plugins.get('task-utils');

    taskUtils.initTask();
    taskUtils.disableInteraction();
    taskUtils.addCleanup(function(){
        $("#task-container").empty();
    })

    $("#task-container").empty();
    $("#task-container").append("<p>Please use a few sentences to describe the virtual world shown to the left.</p>");
    $("#task-container").append("<p><textarea id='annotation' style='width:90%;height:10em;'/></p>");
    $("#task-container").append("<p><button id='task-submit-button'>Submit</button></p>");
    var button = $('#task-submit-button');

    taskUtils.restoreSnapshot(snapshot);

    button.click(function() {
        var text = $("#annotation").val();

        console.log("submitting annotation", text);

        if (!text || text.length < 5) {
            alert("Please enter a longer description!");
            return;
        }

        var result_data = {
            dataStruct: {
                annotation: text
            }
        }

        tracker.submit_task(result_data);
        taskUtils.cleanupTask();
    });

    game.updateDirtyChunks();
}

module.exports = function(tracker, invalidate) {
    tracker.register_game("snapshot_annotate_v1", function(data) {
        console.log('got snapshot game', data)
        data.dataStruct.snapshot.position[0] += 1; // compat with bad camera placement
        data.dataStruct.snapshot.position[1] += 1;
        data.dataStruct.snapshot.position[2] += 1;
        annotationGame(tracker, data.dataStruct.snapshot, invalidate);
    });

    tracker.register_game("snapshot_annotate_v2", function(data) {
        console.log('got snapshot game', data)
        data.dataStaticWorld.snapshot.position[0] += 1; // compat with bad camera placement
        data.dataStaticWorld.snapshot.position[1] += 1;
        data.dataStaticWorld.snapshot.position[2] += 1;
        annotationGame(tracker, data.dataStaticWorld.snapshot, invalidate);
    });

    tracker.register_game("snapshot_annotate_v3", function(data) {
        console.log('got snapshot game', data)
        annotationGame(tracker, data.dataStaticWorld.snapshot, invalidate);
    });
}
