const convnet = require('convnetjs');
const Brain = require('brain.js');
const MatrixLog = require('matrix-log.js');
const gpuMock = require('gpu-mock.js');
// const inject = require('../utilities/inject');
const acorn = require('acorn');
const acornLoose = require('acorn/dist/acorn_loose');
const createQueryWrapper = require('query-ast');

describe('Convolution', () => {
  describe('backpropagation', () => {
    describe('algorithm', () => {
      function getConvLayerInstance(settings) {
        const filters = [];
        for (let i = 0; i < settings.filterCount; i++) {
          const filter = {
            sx: settings.filterWidth,
            sy: settings.filterHeight,
            depth: settings.input.depth,
            dw: [],
            w: []
          };
          filters.push(filter);
        }
        const instance = {
          out_depth: 1,
          in_act: {
            sx: settings.input.width,
            sy: settings.input.height,
            w: [],
            dw: []
          },

          filters: filters,
          biases: {
            dw: [1,2,3,4]
          },
          stride: Math.max(settings.stride, 1),
          pad: settings.padding || 0,
          in_sx: settings.input.width,
          in_sy: settings.input.weight,
          in_depth: settings.input.depth,
          sx: settings.filterWidth,
          sy: settings.filterHeight,
          out_sx: settings.width,
          out_sy: settings.height,
          out_act: {
            get_grad: function(x, y) {
              return 0;
            }
          },
          backwardCallback: settings.backwardCallback
        };
        return instance;
      }
      function getConvLayerBackward() {
        const value = `this.backwardCallback({
          targets: ['f.dw', 'V.dw'],
          x, y, d,
          ox, oy,
          fx, fy, fd,
          ax, ay
        });`;

        const target = 'V.dw[ix1] += f.w[ix2]*chain_grad;';

        const result = convnet.ConvLayer.prototype.backward.toString()
          .replace(target,
            target + '\n' + value + '\n')
          .replace('global.zeros', 'new Array');

        return eval('(' + result + ')');
      }
      describe('filters', () => {
        it('can backpropagate to a 4x4 filter matrix', () => {
          const filterWidth = 4;
          const filterHeight = 4;
          const filterCount = 1;
          const outputWidth = 4;
          const outputHeight = 4;
          const outputDepth = 1;
          const inputWidth = 4;
          const inputHeight = 4;
          const inputDepth = 1;
          const convnetMatrixLog = new MatrixLog('filters', filterWidth, filterHeight);

          getConvLayerBackward().call(
            getConvLayerInstance({
              filterWidth,
              filterHeight,
              filterCount,
              width: outputWidth,
              height: outputHeight,
              depth: outputDepth,
              input: {
                width: inputWidth,
                height: inputHeight,
                depth: inputDepth,
              },
              backwardCallback: (stats) => {
                if (stats.targets && stats.targets.join(',') === 'f.dw,V.dw') {
                  convnetMatrixLog.add('deltas', stats.fx, stats.fy, stats.ox, stats.oy, outputWidth, outputHeight);
                  convnetMatrixLog.add('inputs', stats.fx, stats.fy, stats.ax, stats.ay, inputWidth, inputHeight);
                }
              }
            }));

          // gpuMock(Brain.layers.Convolution.compareFilters)
          console.log(convnetMatrixLog.toString('deltas'));
          // TODO: compare with brain.js
        });
      });
    });
  });
});