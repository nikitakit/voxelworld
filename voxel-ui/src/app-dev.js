var createGameCustom = require('./voxel-engine-wrapper')
var Tracker = require('./tracker');
var preserveForms = require('./preserve-forms');


var lcmproto_ws_bridge = require('lcmproto_ws_bridge')
LCMProto = lcmproto_ws_bridge.LCMProto

var $ = require('jquery');

module.exports = function() {
    preserveForms();

    game = createGameCustom();
    game.plugins.get('task-utils').setDev(true);

    lcm = new LCMProto("ws://localhost:8000");
    lcm.on_ready(function() {
        startLCM(lcm, game)
    });

    $("#teleport-confirm").click(function() {
        game.controls.target().avatar.position.x = $("#px").val();
        game.controls.target().avatar.position.y = $("#py").val();
        game.controls.target().avatar.position.z = $("#pz").val();
    });

    $("#screenshot-confirm").click(function() {
        var data = game.shell.canvas.toDataURL();
        $("#screenshot-target").attr("src", data);
    });

    game.shell.on('tick', function() {
        if (!game.shell.pointerLock) {
            return;
        }
        var pos = game.cameraPosition();
        pos[0] = Math.floor(pos[0]);
        pos[1] = Math.floor(pos[1]);
        pos[2] = Math.floor(pos[2]);
        $("#px").val(pos[0]);
        $("#py").val(pos[1]);
        $("#pz").val(pos[2]);
        if (game.getBlock(pos) != 0) {
            $("#snapshot-warning").show().attr('title', 'WARNING: camera is inside a block');
        } else {
            $("#snapshot-warning").hide();
        }
    })

    return game;
}

function useDynamicChunks() {
    if ($("#noDynamicChunks").prop('checked')) {
        return false;
    }
    return true;
}

function startLCM(lcm, game) {
    game.lcm = lcm

    var toggleChunkLoader = function() {
        if (!useDynamicChunks()) {
            game.chunkLoader.disable();
        } else {
            game.chunkLoader.enable();
        }
    }

    toggleChunkLoader();
    $("#noDynamicChunks").click(toggleChunkLoader);

    game.chunkLoader.onChunkRequested(function(position, dimensions, data) {
        lcm.call("WorldService.GetRegions", {
            position: position,
            dimensions:dimensions
        }, function (msg) {drawRegions(game, msg.regions);})
    });

    var tracker = initTracker(lcm, game);

    $("#path-list").click(function() {
        lcm.call("WorldService.List", {}, function(reply) {updatePaths(game, reply.paths)} );
    });

    $("#path-confirm").click(function() {
        var path = $("#path-selector").val();
        if (path === null) {
            return;
        }
        lcm.call("WorldService.Load", {path:path});
        game.chunkLoader.invalidateAllChunks();
    });

    $("#render-invalidate-confirm").click(function() {
        game.chunkLoader.invalidateAllChunks();
    })

    $("#render-confirm").click(function() {
        var path = $("#path-selector").val();
        if (path === null) {
            return;
        }
        lcm.call("WorldService.GetRegions", {
            position: [$("#x").val(), $("#y").val(), $("#z").val()],
            dimensions: [$("#dx").val(),$("#dy").val(),$("#dz").val()]
        }, function (msg) {drawRegions(game, msg.regions);});
    });

    $(".sidebar-button").click(function(event) {
        var tabName = event.currentTarget.id.slice("show-".length);
        $(".sidebar-tab").hide();
        $("#sidebar-" + tabName).show();
    });

    $(".sidebar-tab").hide();
    $(".sidebar-tab-default").show();

    $("#snapshot-confirm").click(function(){
        takeSnapshot(game);
    });

    $("#snapshot-find-confirm").click(function(){
        lcm.call("SnapshotService.Find", {
            paths: [$("#snapshot-find-folder").val()]
        }, function(msg) {
            if (msg.snapshots && msg.snapshots.length > 0) {
                $("#snapshot-selector").empty();
                for (var i = 0; i < msg.snapshots.length; i++) {
                    addSnapshot(msg.snapshots[i]);
                }
            }
        });
    });

    $("#snapshot-load-confirm").click(function(){
        var snapshot = $('#snapshot-selector').find('option:selected').data('snapshot');
        game.plugins.get('task-utils').restoreSnapshot(snapshot);
    });

    $("#prep-game-confirm").click(function(){
        var game = $('#prep-game-selector').find('option:selected').val();
        var baseTask = {"game": game};
        var snapshot;

        console.log(game);
        if (game == "preview") {
            baseTask.dataStruct = {};
        } else if (game == "snapshot_annotate_v2" || game == "snapshot_annotate_v3") {
            baseTask.dataStaticWorld = {
                    "snapshot": $('#snapshot-selector').find('option:selected').data('snapshot'),
                    "data": {}
                };
        } else if (game == "whereis_annotate_v1") {
            snapshot = $('#snapshot-selector').find('option:selected').data('snapshot')
            baseTask.dataStaticWorld = {
                    "snapshot": snapshot,
                    "data": {
                    }
                };
        } else if (game == "whereis_guess_v1") {
            snapshot = $('#snapshot-selector').find('option:selected').data('snapshot')
            baseTask.dataStaticWorld = {
                    "snapshot": snapshot,
                    "data": {
                        "x": Math.floor(snapshot.position[0]),
                        "y": Math.floor(snapshot.position[1]),
                        "z": Math.floor(snapshot.position[2]),
                        "annotation": "An error occured. Please do not complete this task."
                    }
                };
        } else {
            baseTask.dataStruct = {};
        }

        $("#prep-json").val(JSON.stringify(baseTask));
    });


    $("#prep-confirm").click(function(){
        lcm.call("TaskService.Submit", {
            tasks: [
                getNamedTask()
            ],
            save: false,
            returnProcessed: false,
            activate: true
        });
    });

    $("#prep-save-confirm").click(function(){
        lcm.call("TaskService.Submit", {
            tasks: [
                getNamedTask()
            ],
            save: false,
            returnProcessed: false,
            activate: true
        });
    });

    $("#prep-activate-and-return-confirm").click(function(){
        lcm.call("TaskService.Submit", {
            tasks: [
                getNamedTask()
            ],
            save: false,
            returnProcessed: true,
            activate: true
        }, loadTaskPrep);
    });

    $("#prep-load-confirm").click(function(){
        lcm.call("TaskService.Find", {
            names: [$('#prep-name').val()],
            returnProcessed: false,
            returnResponses: false
        }, loadTaskPrep);
    });

    $("#prep-load-processed-confirm").click(function(){
        lcm.call("TaskService.Find", {
            names: [$('#prep-name').val()],
            returnProcessed: true,
            returnResponses: false
        }, loadTaskPrep);
    });

    $("#prep-load-responses-confirm").click(function(){
        lcm.call("TaskService.Find", {
            names: [$('#prep-name').val()],
            returnProcessed: true,
            returnResponses: false
        }, loadResponses);
    });

    lcm.subscribe("TrackerService.Activate/a/[A-Za-z0-9]+", "voxelproto.task.CrowdTaskResponse", function(msg) {
        $("#prep-submitted-date").val(new Date().toLocaleString());
        $("#prep-submitted-json").val(JSON.stringify(msg));
    });
}

function loadResponses(msg) {
    if (msg.tasks) {
        console.log(msg.tasks);
        for (var i = 0; i < msg.tasks.length; i++) {
            var task = msg.tasks[i];
            if (task.responses && task.responses.length > 0) {
                console.log("Have responses!");
                $("#prep-submitted-date").val("");
                $("#prep-submitted-json").val(JSON.stringify(task.responses));
            }
        }
    }
}

function loadTaskPrep(msg) {
    if (msg.tasks) {
        console.log(msg.tasks);
        for (var i = 0; i < msg.tasks.length; i++) {
            var task = msg.tasks[i];
            if (task.processedTask !== undefined) {
                $("#prep-json").val(JSON.stringify(task.processedTask));
            } else {
                $("#prep-json").val(JSON.stringify(task.baseTask));
            }
        }
    }
}

function getCurrentDateString() {
    var date = new Date();
    return date.toISOString().split('T')[0] + "T" + date.toTimeString().split(' ')[0];
}

function addSnapshot(msg) {
    if (!msg.name) {
        msg.name = "Untitled snapshot";
    }
    var optionElement = $("<option>" + msg.name + "</option>");
    optionElement.data('snapshot', msg);
    $("#snapshot-selector").append(optionElement);
}

function updatePaths(game, paths) {
    console.log("Got paths")
    console.log(paths)
    for (var i = 0; i < paths.length; i++) {
        $("#path-selector").append("<option>" + paths[i] + "</option>")
    }
}

function drawRegions(game, regions) {
    if (regions !== undefined && regions.length > 0 ) {
        for (var i = 0; i < regions.length; i++) {
            game.chunkLoader.drawRegion(regions[i]);
        }
    }
}

function takeSnapshot(game) {
    var position, rotation;
    position = game.cameraPosition();
    var camera = game.cameraPlugin.camera;

    // rotation may not be right, check if any need to be negated
    rotation = [camera.rotationX, camera.rotationY, camera.rotationZ];

    console.log("name is", $("#snapshot-folder").val() + "/" + getCurrentDateString());
    lcm.call("SnapshotService.Save", {
        snapshots: [{
            position: position,
            rotation: rotation,
            name: $("#snapshot-folder").val() + "/" + getCurrentDateString()
        }]
    });
}

function getNamedTask() {
    var name = $('#prep-name').val();

    var data = $("#prep-json").val();
    var baseTask;
    try {
        baseTask = JSON.parse(data);
    } catch (err) {
        alert('Bad JSON format');
        throw err;
    }

    return {
        name: name,
        baseTask: baseTask
    };
}

function initTracker(lcm, game) {
    var tracker = Tracker(lcm);
    game.tracker = tracker;
    tracker.register_game("preview", function(data) {
        console.log("Preview game is run!");
    })

    require("./game-auto-approve")(tracker);
    require('./game-snapshot-annotate')(tracker);
    require('./game-whereis-annotate')(tracker);
    require('./game-whereis-review')(tracker);
    require('./game-whereis-guess')(tracker);

    return tracker;
}
