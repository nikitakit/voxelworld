/*
  LCMProto WebSocket Bridge: javascript client library

  This library imitates the LCM APIs, and fulfills requests by forwarding them
  to a websocket server.

  PUBLIC API:

    new LCMProto(websocket_url)
    on_ready(callback)
    subscribe(channel, msg_type, callback)
    unsubscribe(subscription_id)
    publish(channel, msg_type, msg)

  EXAMPLE CODE:

    var l = new LCM("ws://localhost:8000");

    l.on_ready(function() {
      var sub = l.subscribe("EXAMPLE", "mctest.ExampleMessage", function(msg) {
        alert(msg.timestamp);
      });

      window.setTimeout(function() {
        l.unsubscribe(sub);
      }, 1000);

      l.publish("EXAMPLE", "mctest.ExampleMessage", {
        position: [1, 2, 3],
        name: "bridged",
        enabled: True
      });
    });

  This code is based on the original LCM Websocket bridge, which is part of
  https://github.com/pioneers/forseti2
*/


function LCMProto (ws_uri) {
  // LCM over WebSockets main class
  var self = this;
  this.ready = false;
  this.ws = new WebSocket(ws_uri);
  this.ws.onmessage = function(evt) {
    self.delegate(JSON.parse(evt.data));
  };
  this.callbacks = {}; // indexed by subscription id
  this.active_calls = {}; // indexed by uid
  this.method_fns = {}; // indexed by method name
}

LCMProto.prototype.on_ready = function(cb) {
  // Call the callback once this LCMProto instance is ready to receive commands
  var self = this;
  this.ws.onopen = function () {
      self.ready = true;
      cb(self);
  };
};

LCMProto.prototype.ws_send = function(type, data) {
  // Internal convenience method for sending data over the websocket
  this.ws.send(JSON.stringify({type: type, data:data}));
};

LCMProto.prototype.delegate = function(request) {
  // Internal method that delegates data received to the appropriate handler
  var callback;
  switch(request.type) {
  case "packet":
    callback = this.callbacks[request.data.subscription_id];
    if (callback !== undefined) {
      callback(request.data.msg);
    }
    break;
  case "call_response":
    callback = this.active_calls[request.data.uid];
    if (callback !== undefined) {
      callback(request.data.msg);
    }
    break;
  case "server_request":
    method_fn = this.method_fns[request.data.method];
    if (method_fn !== undefined) {
      method_fn(request.data.msg,
          this.create_responder(request.data.method, request.data.uid));
    }
    break;
  default:
    throw "Invalid request!";
  }
};

LCMProto.prototype.generate_uuid = function() {
  // Internal method to generate unique subscription IDs
  var d = new Date().getTime();
  var uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = (d + Math.random()*16)%16 | 0;
    d = Math.floor(d/16);
    return (c=='x' ? r : (r&0x7|0x8)).toString(16);
  });
  return uuid;
};

LCMProto.prototype.generate_lrpc_uid = function() {
  var uid = "";
  var d = new Date().getTime();

  for (var i=0; i < 24; i++) {
      var r = (d + Math.random()*32)%32 | 0;
      uid = uid + r.toString(32);
  }
  return uid;
};

LCMProto.prototype.subscribe = function(channel, msg_type, callback) {
  // Subscribe to an LCM channel with a callback
  // Unlike the core LCM APIs, this requires a message type, and the
  // callback receives an already-decoded message as JSON instead of
  // an encoded string
  //
  // Invalid requests are silently ignored (there is no error callback)

  var subscription_id = this.generate_uuid();
  this.callbacks[subscription_id] = callback;

  this.ws_send("subscribe", {channel: channel,
                             msg_type: msg_type,
                             subscription_id: subscription_id});
  return subscription_id;
};

LCMProto.prototype.unsubscribe = function(subscription_id) {
  // Unsubscribe from an LCMProto channel, using a subscription id
  //
  // Invalid requests are silently ignored (there is no error callback)

  this.ws_send("unsubscribe", {subscription_id: subscription_id});
  delete this.callbacks[subscription_id];
};

LCMProto.prototype.publish = function(channel, msg_type, data) {
  // Publish a message to an LCMProto channel
  //
  // Invalid requests are silently ignored (there is no error callback)
  this.ws_send("publish", {channel: channel,
                           msg_type: msg_type,
                           data: data});
};

LCMProto.prototype.call = function(method, data, callback) {
  // TODO(nikita): docs
  var uid = this.generate_lrpc_uid();
  if (callback !== null && callback !== undefined) {
      this.active_calls[uid] = callback;
  }

  this.ws_send("call", {method: method,
                             uid: uid,
                             data: data});
};

LCMProto.prototype.server_define = function(method, fn) {
  // TODO(nikita): docs
  this.method_fns[method] = fn;
  this.ws_send("server_define", {method: method});
};

LCMProto.prototype.create_responder = function(method, uid) {
  var res = {};
  var self = this;
  res.send = function(msg) {
    self.ws_send("server_response", {method: method, uid:uid, data:msg});
  };
  return res
};


// Exports for running in a CommonJS environment
if (typeof require !== "undefined") {
  exports.LCMProto = LCMProto;
}
