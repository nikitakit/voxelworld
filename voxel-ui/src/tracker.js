var $ = require('jquery');

function BaseTracker (container) {
    var self = this;

    this.container = $(container);
    this.ready = false;
    this.active_task = null;
    this.callback_on_ready = null;
    this.task_started_time = null;
    this.submitted_task_ids = [];

    this.update_assignments({'loaded': false, assignments:[]});
    this.games = {};
}

BaseTracker.prototype.register_game = function(game, cb) {
    this.games[game] = cb;
}

BaseTracker.prototype.set_ready = function() {
    if (! this.ready) {
        this.ready = true;
        if (this.callback_on_ready !== null) {
            this.callback_on_ready();
        }
    }
}

BaseTracker.prototype.on_ready = function(cb) {
    // Call the callback once this tracker instance is ready to receive commands
    if (this.ready) {
        cb(this);
    } else {
        self.callback_on_ready = cb;
    }
};

BaseTracker.prototype.base_submit_task = function(data) {
    if (this.active_task.task_id && data.task_id === undefined) {
        data.task_id = this.active_task.task_id;
    }
    if (data.code === undefined) {
        data.code = 1;
    }

    data.elapsedTime = (Date.now() - this.task_started_time) / 1000;

    if (this.active_task.task_id) {
        this.submitted_task_ids.push(this.active_task.task_id);
    }

    this.active_task = null;
}

BaseTracker.prototype.base_activate_task = function(data) {
    if (data === null) {
        // No task supplied
        return;
    }

    // Prevent multiple activations of the same task
    if (this.active_task && this.active_task.task_id !== undefined && data.task_id !== undefined) {
        if (this.active_task.task_id == data.task_id) {
            return;
        }
    }
    if (data.task_id !== undefined && $.inArray(data.task_id, this.submitted_task_ids) !== -1) {
        return;
    }

    this.active_task = data;

    var game = data.game;
    var gameFunction = this.games[game];
    if (gameFunction === undefined) {
        console.log("error: bad game " + game);
        alert("An error has occured. Please close the page and return to Mechanical Turk.")
        return;
    }

    var self = this;
    $('#startup-text').hide();
    $('#loading-text').show();
    setTimeout(function() {
        try {
            gameFunction(data);
            self.task_started_time = Date.now();
        } catch (e) {
            $('#loading-text').hide();
            alert("An error has occured. Please close the page and return to Mechanical Turk.");
            throw e;
        }
        $('#loading-text').hide();
    }, 10);
};

BaseTracker.prototype.update_assignments = function(data) {
    var contents = "<h2>Your accepted HITs</h2>";

    if (!data.loaded) {
        contents += "<p>Loading...</p>";
        this.container.html(contents);
        return;
    }

    var assignments = data.assignments;
    var haveDescription = false;
    contents += "<ul>";
    for (var i = 0; i < assignments.length; i++) {
        var assignment = data.assignments[i];
        if (assignment.completed) {
            contents += "<li>[DONE] " + assignment.name + "</li>"
        } else {
            contents += "<li>" + assignment.name + "</li>"
            if (!haveDescription) {
                contents += "<p>" + assignment.description + "</p>";
                haveDescription = true;
            }
        }
    }
    contents += "</ul>";
    this.container.html(contents);
};

// A tracker that automatically starts the preview game, and doesn't require a
// server to work
function PreviewTracker(container) {
    BaseTracker.call(this, container);

    var self = this;

    this.register_game('preview', function(data) {
        self.container.html(contents + "<p>If you see this message, an error has occured. <br/> Please do not accept any assignments. </p>");
    });

    // delay so the preview game can be swapped out
    setTimeout(function(){
        self.update_assignments({
            loaded: true,
            assignments:[{
                completed: false,
                name: "Preview the console",
                description: ""
            }]
        });

        // Delay so that the DOM has time to re-render
        setTimeout(function(){
            self.activate_task({game: 'preview'});
        }, 10);
    }, 100);

    this.set_ready();

    document.title = "[Preview] " + document.title;
}

PreviewTracker.prototype = Object.create(BaseTracker.prototype);

PreviewTracker.prototype.submit_task = function(data) {
    this.base_submit_task(data);
};

PreviewTracker.prototype.activate_task = function(data) {
    this.base_activate_task(data);
};

function url(s) {
    var l = window.location;
    return ((l.protocol === "https:") ? "wss://" : "ws://") + l.host + s;
}

// The tracker used by mturk workers, which needs a workerId and communicates
// with a server to receive work and send progress updates
function WSTracker (container, workerId) {
    BaseTracker.call(this, container);

    var self = this;

    // TODO: error handling for bad worker ID
    this.workerId = workerId;

    this.ws = new WebSocket(url('/ws?workerId=' + this.workerId));
    this.ws.onmessage = function(evt) {
        self.delegate(JSON.parse(evt.data));
    };
    this.callbacks = {}; // indexed by subscription id
    this.outObj = undefined;

    function request_task() {
        if (self.active_task === null) {
            self.ws_send("get_task", {});
        }
    };
    this.request_timer = null;

    this.ws.onopen = function() {
        self.set_ready();
        request_task();
        self.request_timer = setInterval(request_task, 1000);
    }
    this.ws.onclose = function() {
        if (self.request_timer !== null) {
            clearInterval(self.request_timer);
        }
        self.container.html("<h2>Connection lost</h2> Please close or refresh the page");
        alert("Connection to server lost. Please refresh the page.");
        $("#task-container").html("<p>This task is not available without a working connection.</p>");
    }
}

WSTracker.prototype = Object.create(BaseTracker.prototype);

WSTracker.prototype.ws_send = function(type, data) {
  // Internal convenience method for sending data over the websocket
  this.ws.send(JSON.stringify({type: type, data:data}));
};

WSTracker.prototype.delegate = function(request) {
    // Internal method that delegates data received to the appropriate handler
    var callback;
    switch(request.type) {
    case "task":
        this.activate_task(request.data);
        break;
    case "assignment_list":
        this.update_assignments(request.data);
        break;
    case "packet": // unused; but here for compat with lcm websocket bridge
        callback = undefined; // = this.callbacks[request.data.subscription_id];
        if (callback !== undefined) {
            callback(request.data.msg);
        }
        break;
    case "prompt_replace":
        this.prompt_replace();
        break;
    case "prompt_returned":
        this.prompt_returned();
        break;
    case "prompt_assignment_completed":
        this.prompt_assignment_completed();
        break;
    default:
        throw "Invalid request!";
    }
};

WSTracker.prototype.prompt_replace = function() {
    var do_not_replace = confirm("You already have another console open, but only one is allowed.\nPress OK to close this console.\nPress Cancel to close the other console (any partial work there will be discarded)");
    // var do_replace = confirm("You already have another console open, but only one is allowed.\nPress OK to close the other console (any partial work will be discarded)\nPress Cancel to close this console.");
    if  (!do_not_replace) {
        this.ws_send("replace", {});
    } else {
        this.ws.close();
    }
}

WSTracker.prototype.prompt_returned = function() {
    alert("You have returned or abandoned one of our HITs.\nPlease close or refresh this page.\nIf you continue to work on an already-returned assignment, we will not be able to credit your work.");
}

WSTracker.prototype.prompt_assignment_completed = function() {
    alert("Assignment completed!\nYou may return to Mechanical Turk and submit the assignment.\n\nA new task will load in this page as soon as you accept another assignment from us.");
}

WSTracker.prototype.submit_task = function(data) {
    this.base_submit_task(data);
    this.ws_send("submit_task", data);
}

WSTracker.prototype.activate_task = function(data) {
    if (data === null) {
        // No task supplied
        alert('No more tasks to complete! Sign up for more!');
    }

    this.base_activate_task(data);
};

// The tracker used for development, which communicates over LCM to more
// seemlessly interface with the dev-side stack
function LCMTracker (container, lcm) {
    BaseTracker.call(this, container);

    var self = this;

    this.lcm = lcm;

    this.lcm.server_define("TrackerService.Activate", function(req, out) {
        self.outObj = out;
        self.activate_task(req);
    });

    this.set_ready();
}

LCMTracker.prototype = Object.create(BaseTracker.prototype);

LCMTracker.prototype.submit_task = function(data) {
    this.base_submit_task(data);

    var msg = {};
    // Copy over dataStruct and the like.
    // Things like error codes are currently not accomodated, so this strips
    // them
    for (var prop in data) {
      if( !data.hasOwnProperty(prop) ) {
          continue;
      }
      if (prop.startsWith('data') || prop == 'elapsedTime') {
          msg[prop] = data[prop];
      }
    }

    if (this.outObj !== undefined) {
        this.outObj.send(msg);
        this.outObj = undefined;
    }
}

LCMTracker.prototype.activate_task = function(data) {
    if (data === null) {
        // No task supplied
        console.log("Warning: no more tasks to complete!")
    }

    this.base_activate_task(data);
};

module.exports = function(lcm) {
    var queryDict = {};
    location.search.substr(1).split("&").forEach(function(item) {queryDict[item.split("=")[0]] = item.split("=")[1]})

    var tracker;
    if (lcm !== undefined && lcm) {
        // If an LCM instance is passed, we are in dev mode.
        tracker = new LCMTracker('#tracker', lcm);
    } else if (queryDict.preview !== undefined) {
        tracker = new PreviewTracker('#tracker');
    } else if (queryDict.workerId === 'None' || !queryDict.workerId) {
        alert("An error has occured. Please close the page and return to Mechanical Turk.");
        throw new Error("No worker id");
    } else {
        tracker = new WSTracker('#tracker', queryDict.workerId);
    }

    return tracker;
}
