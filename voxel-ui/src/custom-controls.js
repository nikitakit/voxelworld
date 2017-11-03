module.exports = function(game, opts) {
  return new CustomControls(game, opts)
}

module.exports.pluginInfo = {
}

function CustomControls(game, opts) {
    this.game = game;
    this.speed = 0.0112;
    this.collidables = game.controls.target().collidables;

    game.controls.tick = this.tick.bind(this);

    // Removing gravity, as commented out here, still leaves an infinitesimal
    // force in the vertical direction. My hypothesis is that this is due to
    // floating point precision error.
    // Integrated over a long period of time this causes the camera to float
    // upwards, which is not acceptable. Therefore we explicitly zero out all
    // force, acceleration, and velocity terms.
    // game.controls.target().removeForce(game.gravity);
    game.controls.target().forces = [0,0,0];
    game.controls.target().acceleration = [0,0,0];
    game.controls.target().velocity = [0,0,0];

    game.controls.target().collidables = [];
}


CustomControls.prototype.tick = function(dt) {
    if (!this.game.controls._target) {
        return;
    }

    var state = this.game.controls.state;
    var target = this.game.controls._target;
    if (state.forward) {
        target.velocity[2] = -this.speed;
    } else if (state.backward) {
        target.velocity[2] = this.speed;
    } else {
        target.velocity[2] = 0.0;
    }

    if (state.left) {
        target.velocity[0] = -this.speed;
    } else if (state.right) {
        target.velocity[0] = this.speed;
    } else {
        target.velocity[0] = 0.0;
    }

    if (state.jump) {
        target.velocity[1] = this.speed;
    } else if (state.crouch) {
        target.velocity[1] = -this.speed;
    } else {
        target.velocity[1] = 0.0;
    }

    var can_fire = true

    if(state.fire || state.firealt) {
      if(this.game.controls.firing && this.game.controls.needs_discrete_fire) {
        this.game.controls.firing += dt
      } else {
        if(!this.game.controls.fire_rate || floor(this.game.controls.firing / this.game.controls.fire_rate) !== floor((this.game.controls.firing + dt) / this.game.controls.fire_rate)) {
          this.game.controls.onfire(state)
        }
        this.game.controls.firing += dt
      }
    } else {
      this.game.controls.firing = 0
    }


    var x_rotation = state.x_rotation_accum * this.game.controls.rotation_scale
       , y_rotation = state.y_rotation_accum * this.game.controls.rotation_scale
       , z_rotation = state.z_rotation_accum * this.game.controls.rotation_scale
       , pitch_target = this.game.controls._pitch_target
       , yaw_target = this.game.controls._yaw_target
       , roll_target = this.game.controls._roll_target

     if (pitch_target === yaw_target && yaw_target === roll_target) {
       pitch_target.eulerOrder = 'YXZ'
     }

     pitch_target.rotation.x = clamp(pitch_target.rotation.x + clamp(x_rotation, this.game.controls.x_rotation_per_ms), this.game.controls.x_rotation_clamp)
     yaw_target.rotation.y = clamp(yaw_target.rotation.y + clamp(y_rotation, this.game.controls.y_rotation_per_ms), this.game.controls.y_rotation_clamp)
     roll_target.rotation.z = clamp(roll_target.rotation.z + clamp(z_rotation, this.game.controls.z_rotation_per_ms), this.game.controls.z_rotation_clamp)

     if(this.game.controls.listeners('data').length) {
       this.game.controls.emitUpdate()
     }

     state.x_rotation_accum =
     state.y_rotation_accum =
     state.z_rotation_accum = 0
}

var max = Math.max
  , min = Math.min
  , sin = Math.sin
  , abs = Math.abs
  , floor = Math.floor
  , PI = Math.PI

function clamp(value, to) {
  return isFinite(to) ? max(min(value, to), -to) : value
}
