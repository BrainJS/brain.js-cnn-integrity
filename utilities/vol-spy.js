const MatrixLog = require('matrix-log.js');

// Note: this is a VolSpy, which isn't intended to give values, but rather just allows us to see how a Vol is used
function VolSpy(sx, sy, depth) {
  this.sx = sx;
  this.sy = sy;
  this.depth = depth;
  this.matrixLog = null;
}

VolSpy.prototype = {
  passive_x: function() {
    this.w = [];
  },
  passive_dw: function() {
    this.dw = [];
  },
  spy_w: function(matrixLog, name) {
    this.w = new Proxy([], (index) => {
      const point = VolSpy.pointFromIndex(this.sx, this.sy, this.depth, index);
      matrixLog.add(name, )
    });
  },
  spy_dw: function() {},
  spy_get_grad: function() {}
};

VolSpy.pointFromIndex = function pointFromIndex(width, height, depth, index) {
  // NOTE: simply math maybe?
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let z = 0; z < depth; z++) {
        const point = VolSpy.indexFromPoint(width, height, depth, x, y, z);
        if (point === index) {
          return {
            x, y, z
          };
        }
      }
    }
  }

  throw new Error(`could not find point ${ index }`);
};

VolSpy.indexFromPoint = function indexFromPoint(width, height, depth, x, y, z) {
  // Copied from inside convnetjs
  return (((width * y) + x) * depth) + z;
};

module.exports = VolSpy;