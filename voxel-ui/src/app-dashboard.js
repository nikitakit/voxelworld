var createGameCustom = require('./voxel-engine-wrapper')
var Tracker = require('./tracker');
var $ = require('jquery');

var previewRegion = {
  "voxelsU32": {
    "voxelsU32": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 18, 18, 0, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 18, 0, 0, 0, 0, 18, 0, 0, 0, 0, 17, 0, 0, 0, 0, 17, 0, 0, 18, 18, 17, 18, 18, 18, 18, 17, 18, 18, 0, 18, 17, 18, 0, 0, 18, 18, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 18, 18, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 18, 18, 0, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  },
  "dimensions": [5,6,5],
  "position": [0,0,0],
};

module.exports = function() {
    var game = createGameCustom();
    game.plugins.get('task-utils').setDev(false);

    game.stitcher.on('addedAll', function(){
        var tracker = Tracker();
        tracker.register_game("preview", function(data) {
            game.shell.pointerLock = false;
            game.chunkLoader.drawRegion(previewRegion);
            var camera = game.cameraPlugin.camera;

            camera.position = [-15, -4.5, -15];

            camera.rotationX = 0.25;
            camera.rotationY = -0.80;
            camera.rotationZ = 0;

            $("#task-container").empty();
            $("#task-container").append("<h3>About</h3>");
            $("#task-container").append("<p>You should see a blocky tree to the left of this text.</p>");
            $("#task-container").append("<p>Our tasks will ask you to describe and interact with virtual worlds in the style shown to the left.</p>");
            $("#task-container").append("<br/>");
            $("#task-container").append("<p><strong>Do not accept this task if you cannot see a tree to the left</strong>. This likely means that your browser is unsupported. We recommend that you use Chrome to complete our tasks.</p>");
        });

        require('./game-auto-approve')(tracker);
        require('./game-snapshot-annotate')(tracker);
        require('./game-whereis-annotate')(tracker);
        require('./game-whereis-guess')(tracker);
    });
}
