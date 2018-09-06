const convnet = require('convnetjs');
const compareFilters = require('brain.js/dist/layer/convolution').compareFilters;
const MatrixLog = require('matrix-log.js');
const gpuMock = require('gpu-mock.js');

describe('Convolution', () => {
  describe('backpropagation', () => {
    describe('algorithm shape', () => {
      function getConvNetConvLayerInstance(settings) {
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
          stride: Math.max(settings.stride || 0, 1),
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
          callback: settings.callback
        };
        return instance;
      }
      function getConvNetConvLayerBackward() {
        const value = `this.callback({
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

        return eval(`(${result})`);
      }
      function getBrainConvolutionLayerCompareFilters(settings) {
        const target = 'sum += deltas[this.thread.z][inputY + y][inputX + x] * inputs[this.thread.z][y][x]';
        const result = compareFilters.toString()
          .replace(target, `\nthis.constants.callback({
            deltaX: inputX + x,
            deltaY: inputY + y,
            filterX: this.thread.x,
            filterY: this.thread.y
          })\n`);

        const mockInput = [
          [
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]
          ]
        ];

        return gpuMock(eval(`(${result})`), {
          output: [settings.filterWidth, settings.filterHeight],
          constants: {
            strideX: Math.max(settings.stride || 0, 1),
            strideY: Math.max(settings.stride || 0, 1),
            paddingX: Math.max(settings.padding || 0, 0),
            paddingY: Math.max(settings.padding || 0, 0),
            filterWidth: settings.filterWidth,
            filterHeight: settings.filterHeight,
            filterCount: settings.filterCount,
            callback: settings.callback
          }
        })(mockInput,mockInput,mockInput);
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
          const brainMatrixLog = new MatrixLog('filters', filterWidth, filterHeight);

          const settings = {
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
          };

          getConvNetConvLayerBackward().call(
            getConvNetConvLayerInstance(Object.assign({
              callback: (stats) => {
                if (stats.targets && stats.targets.join(',') === 'f.dw,V.dw') {
                  convnetMatrixLog.add('deltas', stats.fx, stats.fy, stats.ox, stats.oy, outputWidth, outputHeight);
                }
              }
            }, settings)));

          getBrainConvolutionLayerCompareFilters(Object.assign({
            callback: (stats) => {
              brainMatrixLog.add('deltas', stats.filterX, stats.filterY, stats.deltaX, stats.deltaY, outputWidth, outputHeight);
            }
          }, settings));

          const expected = convnetMatrixLog.toString('deltas').split(/\n/g);
          const result = brainMatrixLog.toString('deltas').split(/\n/g);
          expect(result).toEqual(expected);
        });
      });
    });
  });
});