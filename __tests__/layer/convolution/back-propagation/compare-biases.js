const convnet = require('convnetjs');
const compareBiases = require('brain.js/dist/layer/convolution').compareBiases;
const MatrixLog = require('matrix-log.js');
const gpuMock = require('gpu-mock.js');
const utils = require('../../../../utils');

const shortenResults = false;

describe('layer.Convolution.compareBiases()', () => {
  function getConvNetConvLayerInstance(settings) {
    const filters = [];
    for (let i = 0; i < settings.depth; i++) {
      const filter = utils.fillPlusPlusVol(settings.filterWidth, settings.filterHeight, settings.input.depth);
      filters.push(filter);
    }
    const outAct = utils.fillPlusPlusVol(settings.width, settings.height, settings.depth);
    const biases = utils.fillPlusPlusVol(1, 1, settings.depth);

    const instance = {
      in_act: utils.fillPlusPlusVol(settings.input.width, settings.input.height, settings.input.depth),

      filters: filters,
      biases: biases,
      stride: Math.max(settings.stride || 0, 1),
      pad: settings.padding || 0,
      in_sx: settings.input.width,
      in_sy: settings.input.height,
      in_depth: settings.input.depth,
      sx: settings.filterWidth,
      sy: settings.filterHeight,
      out_sx: settings.width,
      out_sy: settings.height,
      out_depth: settings.depth,
      out_act: outAct,
      callback: settings.callback
    };
    return instance;
  }
  describe('algorithm shape', () => {
    function getConvNetConvLayerBackward() {
      const value = `this.callback({ ax,ay,d });`;

      const target = 'this.biases.dw[d] += chain_grad;';

      const backwardString = convnet.ConvLayer.prototype.backward.toString();
      if (backwardString.indexOf(target) < 0) {
        throw new Error(`function injection target of "${target}" cannot be found`);
      }

      const result = backwardString
        .replace(target,
          target + '\n' + value + '\n')
        .replace('global.zeros', 'new Array');

      return eval(`(${result})`);
    }
    function getBrainConvolutionLayerCompareBiases(settings) {
      const target = 'sum += deltas[this.thread.z][y][x];';
      const compareBiasesString = compareBiases.toString();
      if (compareBiasesString.indexOf(target) < 0) {
        throw new Error(`function injection target of "${target}" cannot be found`);
      }
      const compareBiasesInjectedString = compareBiasesString
        .replace(target, target + `\nthis.constants.callback({
          deltaX: x,
          deltaY: y,
          deltaZ: this.thread.z,
          z: this.thread.z
        })\n`);
      const compareBiasesInjected = eval(`(${compareBiasesInjectedString})`);
      const biasDeltas = utils.fillPlusPlus(1, 1, settings.depth);
      const deltas = utils.fillPlusPlus(settings.width, settings.height, settings.depth);

      return gpuMock(compareBiasesInjected, {
        output: [1, 1, settings.depth],
        constants: {
          deltaWidth: settings.width,
          deltaHeight: settings.height,
          callback: settings.callback
        }
      })(biasDeltas, deltas);
    }
    function setupLogs(settings) {
      const convnetMatrixLog = new MatrixLog('biases', 1, 1, settings.depth);
      const brainMatrixLog = new MatrixLog('biases', 1, 1, settings.depth);
      getConvNetConvLayerBackward().call(
        getConvNetConvLayerInstance(Object.assign({
          callback: (stats) => {
            convnetMatrixLog
              .at({
                x: 0,
                y: 0,
                z: stats.d
              });

            // in `backward` called out_act, or chain_grad
            const deltasLog = {
              name: 'deltas',
              x: stats.ax,
              y: stats.ay,
              z: stats.d,
              width: settings.width,
              height: settings.height,
              depth: settings.depth
            };

            convnetMatrixLog.add(deltasLog);
          }
        }, settings)));

      getBrainConvolutionLayerCompareBiases(Object.assign({
        callback: (stats) => {
          if (stats.deltaX < 0) throw new Error('deltaX less than 0');
          if (stats.deltaX > settings.width) throw new Error(`deltaX greater than ${settings.width}`);
          if (stats.deltaY < 0) throw new Error('deltaY less than 0');
          if (stats.deltaY > settings.height) throw new Error(`deltaY greater than ${settings.height}`);

          brainMatrixLog
            .at({
              x: 0,
              y: 0,
              z: stats.z
            });
          brainMatrixLog
            .add({
              name: 'deltas',
              x: stats.deltaX,
              y: stats.deltaY,
              z: stats.deltaZ,
              width: settings.width,
              height: settings.height,
              depth: settings.depth
            });
        }
      }, settings));

      return { convnetMatrixLog, brainMatrixLog };
    }

    it('can backpropagate from a "4x4x1 input matrix" and a "1x1x1 output matrix" to a "2x2x1 filter matrix"', () => {
      const settings = {
        width: 1,
        height: 1,
        depth: 1,
        filterWidth: 2,
        filterHeight: 2,
        input: {
          width: 4,
          height: 4,
          depth: 1,
        },
      };

      const logs = setupLogs(settings);
      const resultDeltas = logs.brainMatrixLog.toString('deltas').split(/\n/g);
      const expectedDeltas = logs.convnetMatrixLog.toString('deltas').split(/\n/g);
      if (shortenResults) {
        resultDeltas.length = 200;
        expectedDeltas.length = 200;
      }
      expect(resultDeltas).toEqual(expectedDeltas);
    });
    it('can backpropagate from a "4x4x1 input matrix" and a "2x2x1 output matrix" to a "2x2x1 filter matrix"', () => {
      const settings = {
        width: 2,
        height: 2,
        depth: 1,
        filterWidth: 2,
        filterHeight: 2,
        input: {
          width: 4,
          height: 4,
          depth: 1,
        },
      };

      const logs = setupLogs(settings);
      const resultDeltas = logs.brainMatrixLog.toString('deltas').split(/\n/g);
      const expectedDeltas = logs.convnetMatrixLog.toString('deltas').split(/\n/g);
      if (shortenResults) {
        resultDeltas.length = 200;
        expectedDeltas.length = 200;
      }
      expect(resultDeltas).toEqual(expectedDeltas);
    });
    it('can backpropagate from a "4x4x1 input matrix" and a "4x4x1 output matrix" to a "4x4x1 filter matrix"', () => {
      const settings = {
        width: 4,
        height: 4,
        depth: 1,
        filterWidth: 4,
        filterHeight: 4,
        input: {
          width: 4,
          height: 4,
          depth: 1,
        },
      };

      const logs = setupLogs(settings);
      const resultDeltas = logs.brainMatrixLog.toString('deltas').split(/\n/g);
      const expectedDeltas = logs.convnetMatrixLog.toString('deltas').split(/\n/g);
      if (shortenResults) {
        resultDeltas.length = 200;
        expectedDeltas.length = 200;
      }
      expect(resultDeltas).toEqual(expectedDeltas);
    });
    it('can backpropagate from a "6x6x8 input matrix" and a "24x24x8 output matrix" to a "10x10x8 filter matrix" with padding of 2 and stride of 2', () => {
      const settings = {
        width: 24,
        height: 24,
        depth: 8,
        filterWidth: 10,
        filterHeight: 10,
        input: {
          width: 6,
          height: 6,
          depth: 8,
        },
        padding: 2,
        stride: 2
      };

      const logs = setupLogs(settings);
      const resultDeltas = logs.brainMatrixLog.toString('deltas').split(/\n/g);
      const expectedDeltas = logs.convnetMatrixLog.toString('deltas').split(/\n/g);
      if (shortenResults) {
        resultDeltas.length = 200;
        expectedDeltas.length = 200;
      }
      expect(resultDeltas).toEqual(expectedDeltas);
    });
    it('can backpropagate from a "12x12x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix" with padding of 2', () => {
      const settings = {
        width: 24,
        height: 24,
        depth: 8,
        filterWidth: 5,
        filterHeight: 5,
        input: {
          width: 12,
          height: 12,
          depth: 8,
        },
        padding: 2,
        stride: 2
      };

      const logs = setupLogs(settings);
      const resultDeltas = logs.brainMatrixLog.toString('deltas').split(/\n/g);
      const expectedDeltas = logs.convnetMatrixLog.toString('deltas').split(/\n/g);
      if (shortenResults) {
        resultDeltas.length = 200;
        expectedDeltas.length = 200;
      }
      expect(resultDeltas).toEqual(expectedDeltas);
    });
    it('can backpropagate from a "24x24x1 input matrix" and a "24x24x8 output matrix" to a "5x5x1 filter matrix" with padding of 2', () => {
      const settings = {
        width: 24,
        height: 24,
        depth: 8,
        filterWidth: 5,
        filterHeight: 5,
        input: {
          width: 24,
          height: 24,
          depth: 1,
        },
        padding: 2
      };

      const logs = setupLogs(settings);
      const resultDeltas = logs.brainMatrixLog.toString('deltas').split(/\n/g);
      const expectedDeltas = logs.convnetMatrixLog.toString('deltas').split(/\n/g);
      if (shortenResults) {
        resultDeltas.length = 200;
        expectedDeltas.length = 200;
      }
      expect(resultDeltas).toEqual(expectedDeltas);
    });
    it('can backpropagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix" with padding of 2', () => {
      const settings = {
        width: 12,
        height: 12,
        depth: 8,
        filterWidth: 5,
        filterHeight: 5,
        input: {
          width: 24,
          height: 24,
          depth: 1,
        },
        padding: 2
      };

      const logs = setupLogs(settings);
      const resultDeltas = logs.brainMatrixLog.toString('deltas').split(/\n/g);
      const expectedDeltas = logs.convnetMatrixLog.toString('deltas').split(/\n/g);
      if (shortenResults) {
        resultDeltas.length = 200;
        expectedDeltas.length = 200;
      }
      expect(resultDeltas).toEqual(expectedDeltas);
    });
    it('can backpropagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix" with padding of 2 and stride of 2', () => {
      const settings = {
        width: 12,
        height: 12,
        depth: 8,
        filterWidth: 5,
        filterHeight: 5,
        input: {
          width: 24,
          height: 24,
          depth: 1,
        },
        padding: 2,
        stride: 2
      };

      const logs = setupLogs(settings);
      const resultDeltas = logs.brainMatrixLog.toString('deltas').split(/\n/g);
      const expectedDeltas = logs.convnetMatrixLog.toString('deltas').split(/\n/g);
      if (shortenResults) {
        resultDeltas.length = 200;
        expectedDeltas.length = 200;
      }
      expect(resultDeltas).toEqual(expectedDeltas);
    });
  });
  describe('output', () => {
    function setupCompareBiases(settings) {
      const stride = Math.max(settings.stride || 0, 1);
      const padding = Math.max(settings.padding || 0, 0);
      const paddedInputWidth = settings.input.width + padding;
      const paddedInputHeight = settings.input.height + padding;
      const slideWidth = Math.min(settings.width, paddedInputWidth);
      const slideHeight = Math.min(settings.height, paddedInputHeight);

      return gpuMock(compareBiases, {
        output: [1, 1, settings.depth],
        constants: {
          strideX: stride,
          strideY: stride,
          paddingX: padding,
          paddingY: padding,
          filterWidth: settings.filterWidth,
          filterHeight: settings.filterHeight,
          inputWidth: settings.input.width,
          inputHeight: settings.input.height,
          deltaWidth: settings.width,
          deltaHeight: settings.height,
          deltaDepth: settings.depth,
          slideWidth: slideWidth,
          slideHeight: slideHeight
        }
      });
    }
    function setupOutputs(settings) {
      const convnetInstance = getConvNetConvLayerInstance(settings);
      const biasDeltas = utils.fillPlusPlus(1, 1, settings.depth);
      const deltas = utils.fillPlusPlus(settings.width, settings.height, settings.depth);

      expect(biasDeltas).toEqual(utils.volDWToArrays(convnetInstance.biases));
      expect(deltas).toEqual((utils.volDWToArrays(convnetInstance.out_act)));

      convnet.ConvLayer.prototype.backward.call(convnetInstance);
      const expected = utils.volDWToArrays(convnetInstance.biases);
      const result = setupCompareBiases(settings)(biasDeltas, deltas);
      return {
        expected, result
      }
    }
    it('can backpropagate from a "4x4x1 input matrix" and a "1x1x1 output matrix" to a "2x2x1 filter matrix"', () => {
      const settings = {
        width: 1,
        height: 1,
        depth: 1,
        filterWidth: 2,
        filterHeight: 2,
        input: {
          width: 4,
          height: 4,
          depth: 1,
        },
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.depth);
      expect(result.length).toBe(settings.depth);
      expect(result).toEqual(expected);
    });
    it('can backpropagate from a "4x4x1 input matrix" and a "2x2x1 output matrix" to a "2x2x1 filter matrix"', () => {
      const settings = {
        width: 2,
        height: 2,
        depth: 1,
        filterWidth: 2,
        filterHeight: 2,
        input: {
          width: 4,
          height: 4,
          depth: 1,
        },
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.depth);
      expect(result.length).toBe(settings.depth);
      expect(result).toEqual(expected);
    });
    it('can backpropagate from a "4x4x1 input matrix" and a "4x4x1 output matrix" to a "4x4x1 filter matrix"', () => {
      const settings = {
        width: 4,
        height: 4,
        depth: 1,
        filterWidth: 4,
        filterHeight: 4,
        input: {
          width: 4,
          height: 4,
          depth: 1,
        },
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.depth);
      expect(result.length).toBe(settings.depth);
      expect(result).toEqual(expected);
    });
    it('can backpropagate from a "6x6x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix" with padding of 2 and stride of 2', () => {
      const settings = {
        width: 24,
        height: 24,
        depth: 8,
        filterWidth: 5,
        filterHeight: 5,
        input: {
          width: 6,
          height: 6,
          depth: 8,
        },
        padding: 2,
        stride: 2
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.depth);
      expect(result.length).toBe(settings.depth);
      expect(result).toEqual(expected);
    });
    it('can backpropagate from a "12x12x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix" with padding of 2', () => {
      const settings = {
        width: 24,
        height: 24,
        depth: 8,
        filterWidth: 5,
        filterHeight: 5,
        input: {
          width: 12,
          height: 12,
          depth: 8,
        },
        padding: 2,
        stride: 2
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.depth);
      expect(result.length).toBe(settings.depth);
      expect(result).toEqual(expected);
    });
    it('can backpropagate from a "24x24x1 input matrix" and a "24x24x8 output matrix" to a "5x5x1 filter matrix" with padding of 2', () => {
      const settings = {
        width: 24,
        height: 24,
        depth: 8,
        filterWidth: 5,
        filterHeight: 5,
        input: {
          width: 24,
          height: 24,
          depth: 1,
        },
        padding: 2
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.depth);
      expect(result.length).toBe(settings.depth);
      expect(result).toEqual(expected);
    });
    it('can backpropagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix" with padding of 2', () => {
      const settings = {
        width: 12,
        height: 12,
        depth: 8,
        filterWidth: 5,
        filterHeight: 5,
        input: {
          width: 24,
          height: 24,
          depth: 1,
        },
        padding: 2
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.depth);
      expect(result.length).toBe(settings.depth);
      expect(result).toEqual(expected);
    });
    it('can backpropagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix" with padding of 2 and stride of 2', () => {
      const settings = {
        width: 12,
        height: 12,
        depth: 8,
        filterWidth: 5,
        filterHeight: 5,
        input: {
          width: 24,
          height: 24,
          depth: 1,
        },
        padding: 2,
        stride: 2
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.depth);
      expect(result.length).toBe(settings.depth);
      expect(result).toEqual(expected);
    });
  });
});