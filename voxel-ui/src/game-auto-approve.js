var $ = require('jquery');

module.exports = function(tracker) {
    tracker.register_game("auto_approve", function(data) {
        console.log("got autoapprove game");
        var button = $('#auto-approve-button');
        if (button.length == 0) {
            $("#task-container").html("<button id='auto-approve-button'>Click to continue</button>");
            var button = $('#auto-approve-button');
        }

        button.click(function(){
            console.log("button pressed");
            tracker.submit_task({});
            button.remove();
        })
    });
}
