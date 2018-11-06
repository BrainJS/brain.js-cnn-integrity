const convnet = require('convnetjs');
const compareFilterDeltas = require('brain.js/dist/layer/fully-connected').compareFilterDeltas;
const MatrixLog = require('matrix-log.js');
const gpuMock = require('gpu-mock.js');
const utils = require('../../../../utils');

const shortenResults = false;

describe('layer.FullyConnected.compareFilterDeltas()', () => {
  function getConvNetConvLayerInstance(settings) {
    const filters = [];
    for (let i = 0; i < settings.height; i++) {
      const filter = utils.fillPlusPlusVol(1, 1, settings.filterHeight * settings.filterWidth);
      filters.push(filter);
    }
    const outAct = utils.fillPlusPlusVol(1, 1, settings.height);
    const biases = utils.fillPlusPlusVol(1, 1, settings.height);
    const instance = {
      in_act: utils.fillPlusPlusVol(settings.input.width, settings.input.height),
      filters,
      biases,
      num_inputs: settings.input.width * settings.input.height,
      out_depth: settings.height,
      out_act: outAct,
      callback: settings.callback
    };
    return instance;
  }
  describe('algorithm shape', () => {
    function getConvNetConvLayerBackward() {
      const value = `this.callback({
        i, d
      });`;

      const target = 'tfi.dw[d] += V.w[d]*chain_grad; // grad wrt params';

      const backwardString = convnet.FullyConnLayer.prototype.backward.toString();
      if (backwardString.indexOf(target) < 0) {
        throw new Error(`function injection target of "${target}" cannot be found`);
      }

      const result = backwardString
        .replace(target,
          target + '\n' + value + '\n')
        .replace('global.zeros', 'new Array');

      return eval(`(${result})`);
    }
    function getBrainConvolutionLayerCompareFilterDeltas(settings) {
      const target = 'return filterDeltas[this.thread.y][this.thread.x] + inputWeights[this.thread.y][this.thread.x] * deltas[this.constants.deltaY][this.constants.deltaX];';
      const compareFilterDeltasString = compareFilterDeltas.toString();
      if (compareFilterDeltasString.indexOf(target) < 0) {
        throw new Error(`function injection target of "${target}" cannot be found`);
      }
      const compareFilterDeltasInjectedString = compareFilterDeltasString
        .replace(target, `\nthis.constants.callback({
          inputX: this.thread.x,
          inputY: this.thread.y,
          deltaX: this.constants.deltaX,
          deltaY: this.constants.deltaY,
          filterX: this.thread.x,
          filterY: this.thread.y
        })\n` + target);
      const compareFilterDeltasInjected = eval(`(${compareFilterDeltasInjectedString})`);
      const filterDeltas = utils.fillPlusPlus(settings.input.width, settings.input.height);
      const inputs = utils.fillPlusPlus(settings.input.width, settings.input.height);
      const deltas = utils.fillPlusPlus(settings.width, settings.height);

      for (let i = 0; i < settings.height; i++) {
        gpuMock(compareFilterDeltasInjected, {
          output: [settings.filterWidth, settings.filterHeight],
          constants: {
            inputWidth: settings.input.width,
            inputHeight: settings.input.height,
            deltaWidth: settings.width,
            deltaHeight: settings.height,
            callback: settings.callback,
            deltaY: i,
            deltaX: 0
          }
        })(filterDeltas, inputs, deltas);
      }
    }
    function setupLogs(settings) {
      const convnetMatrixLog = new MatrixLog('filters', settings.filterWidth, settings.filterHeight);
      const brainMatrixLog = new MatrixLog('filters', settings.filterWidth, settings.filterHeight);
      getConvNetConvLayerBackward().call(
        getConvNetConvLayerInstance(Object.assign({
          callback: (stats) => {
            const inputLocation = utils.lookupXYZ(stats.d, settings.filterWidth, settings.filterHeight);
            convnetMatrixLog
              .at({
                x: inputLocation.x,
                y: inputLocation.y
              });

            // in `backward` called in_act, V, or V.w
            const inputsLog = {
              name: 'inputs',
              x: inputLocation.x,
              y: inputLocation.y,
              width: settings.input.width,
              height: settings.input.height
            };

            // in `backward` called out_act, or chain_grad
            const deltasLog = {
              name: 'deltas',
              x: 0,
              y: stats.i,
              width: settings.width,
              height: settings.height,
            };

            convnetMatrixLog.add(inputsLog);
            convnetMatrixLog.add(deltasLog);
          }
        }, settings)));

      getBrainConvolutionLayerCompareFilterDeltas(Object.assign({
        callback: (stats) => {
          // if (stats.filterX < 0) throw new Error('filterX less than 0');
          // if (stats.filterX > settings.filterWidth) throw new Error(`filterX greater than ${settings.filterWidth}`);
          // if (stats.filterY < 0) throw new Error('filterY less than 0');
          // if (stats.filterY > settings.filterHeight) throw new Error(`filterY greater than ${settings.filterHeight}`);

          // if (stats.deltaX < 0) throw new Error('deltaX less than 0');
          // if (stats.deltaX > settings.width) throw new Error(`deltaX greater than ${settings.width}`);
          // if (stats.deltaY < 0) throw new Error('deltaY less than 0');
          // if (stats.deltaY > settings.height) throw new Error(`deltaY greater than ${settings.height}`);

          if (stats.inputX < 0) throw new Error('inputX less than 0');
          if (stats.inputX > settings.input.width) throw new Error(`inputX greater than ${settings.input.width}`);
          if (stats.inputY < 0) throw new Error('inputY less than 0');
          if (stats.inputY > settings.input.height) throw new Error(`inputY greater than ${settings.input.height}`);

          brainMatrixLog
            .at({
              x: stats.filterX,
              y: stats.filterY
            });
          brainMatrixLog
            .add({
              name: 'inputs',
              x: stats.inputX,
              y: stats.inputY,
              width: settings.input.width,
              height: settings.input.height
            });
          brainMatrixLog
            .add({
              name: 'deltas',
              x: stats.deltaX,
              y: stats.deltaY,
              width: settings.width,
              height: settings.height
            });
        }
      }, settings));

      return { convnetMatrixLog, brainMatrixLog };
    }
    describe('from inputs', () => {
      it('can back propagate from a "4x4x1 input matrix" and a "1x4 output matrix" to a "16x1 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 16,
          input: {
            width: 4,
            height: 4
          },
          filterWidth: 4,
          filterHeight: 4
        };

        const logs = setupLogs(settings);
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can back propagate from a "4x4x1 input matrix" and a "2x2x1 output matrix" to a "2x2x1 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 4,
          input: {
            width: 4,
            height: 4
          },
          filterWidth: 4,
          filterHeight: 4
        };

        const logs = setupLogs(settings);
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can back propagate from a "4x4x1 input matrix" and a "3x3x1 output matrix" to a "9x1 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 9,
          input: {
            width: 4,
            height: 4
          },
          filterWidth: 4,
          filterHeight: 4
        };

        const logs = setupLogs(settings);
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can back propagate from a "4x4x1 input matrix" and a "4x4x1 output matrix" to a "16x1 filter matrix"', () => {
        const settings = {
          width: 4,
          height: 4,
          input: {
            width: 4,
            height: 4
          },
          filterWidth: 4,
          filterHeight: 4
        };

        const logs = setupLogs(settings);
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can back propagate from a "6x6x8 input matrix" and a "24x24x8 output matrix" to a "10x10x8 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 24,
          input: {
            width: 6,
            height: 6
          },
          filterWidth: 6,
          filterHeight: 6
        };

        const logs = setupLogs(settings);
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can back propagate from a "12x12x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 6,
          input: {
            width: 12,
            height: 12
          },
          filterWidth: 12,
          filterHeight: 12
        };

        const logs = setupLogs(settings);
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can back propagate from a "24x24x1 input matrix" and a "24x24x8 output matrix" to a "5x5x1 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 24,
          input: {
            width: 24,
            height: 24
          },
          filterWidth: 24,
          filterHeight: 24
        };

        const logs = setupLogs(settings);
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can back propagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 12,
          input: {
            width: 24,
            height: 24
          },
          filterWidth: 24,
          filterHeight: 24
        };

        const logs = setupLogs(settings);
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can back propagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix"', () => {
        const settings = {
          width: 12,
          height: 12,
          input: {
            width: 24,
            height: 24,
          },
          filterWidth: 24,
          filterHeight: 24
        };

        const logs = setupLogs(settings);
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
    });
    describe('from deltas', () => {
      it('can back propagate from a "4x4x1 input matrix" and a "3x3x1 output matrix" to a "2x2x1 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 9,
          input: {
            width: 4,
            height: 4
          },
          filterWidth: 4,
          filterHeight: 4
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
      it('can back propagate from a "4x4x1 input matrix" and a "2x2x1 output matrix" to a "2x2x1 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 4,
          input: {
            width: 4,
            height: 4
          },
          filterWidth: 4,
          filterHeight: 4
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
      it('can back propagate from a "4x4x1 input matrix" and a "4x4x1 output matrix" to a "4x4x1 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 16,
          input: {
            width: 4,
            height: 4
          },
          filterWidth: 4,
          filterHeight: 4
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
      it('can back propagate from a "6x6x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 24,
          input: {
            width: 6,
            height: 6
          },
          filterWidth: 6,
          filterHeight: 6
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
      it('can back propagate from a "12x12x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 24,
          input: {
            width: 12,
            height: 12
          },
          filterWidth: 12,
          filterHeight: 12
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
      it('can back propagate from a "24x24x1 input matrix" and a "24x24x8 output matrix" to a "5x5x1 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 24,
          input: {
            width: 24,
            height: 24
          },
          filterWidth: 24,
          filterHeight: 24
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
      it('can back propagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 12,
          input: {
            width: 24,
            height: 24
          },
          filterWidth: 24,
          filterHeight: 24
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
      it('can back propagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix"', () => {
        const settings = {
          width: 1,
          height: 12,
          input: {
            width: 24,
            height: 24
          },
          filterWidth: 24,
          filterHeight: 24
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
  });
  describe('output', () => {
    function setupCompareFilterDeltas(settings, i) {
      return gpuMock(compareFilterDeltas, {
        output: [settings.filterWidth, settings.filterHeight],
        constants: {
          filterWidth: settings.filterWidth,
          filterHeight: settings.filterHeight,
          inputWidth: settings.input.width,
          inputHeight: settings.input.height,
          deltaWidth: settings.width,
          deltaHeight: settings.height,
          deltaDepth: settings.depth,
          deltaY: i,
          deltaX: 0
        }
      });
    }
    function setupOutputs(settings) {
      const convnetInstance = getConvNetConvLayerInstance(settings);
      const filterDeltas = utils.fillPlusPlus(settings.filterWidth, settings.filterHeight);
      const inputs = utils.fillPlusPlus(settings.input.width, settings.input.height);
      const deltas = utils.fillPlusPlus(settings.width, settings.height);

      for (let i = 0; i < settings.height; i++) {
        expect(filterDeltas).toEqual(filterDepthHack2D(convnetInstance.filters[i], settings));
      }

      expect(inputs).toEqual(utils.volWToArrays(convnetInstance.in_act)[0]);
      expect(deltas).toEqual(utils.volDWToArrays(convnetInstance.out_act).map(v => v[0]));

      convnet.FullyConnLayer.prototype.backward.call(convnetInstance);
      const expected = convnetInstance.filters.map(filter => filterDepthHack2D(filter, settings));
      const result = [];
      for (let i = 0; i < settings.height; i++) {
        result.push(setupCompareFilterDeltas(settings, i)(filterDeltas, inputs, deltas));
      }
      return {
        expected, result
      }
    }
    it('can back propagate from a "4x4x1 input matrix" and a "1x1x1 output matrix" to a "2x2x1 filter matrix"', () => {
      const settings = {
        width: 1,
        height: 1,
        input: {
          width: 4,
          height: 4,
        },
        filterWidth: 4,
        filterHeight: 4
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.height);
      expect(result.length).toBe(settings.height);
      expect(result).toEqual(expected);
    });
    it('can back propagate from a "4x4x1 input matrix" and a "2x2x1 output matrix" to a "2x2x1 filter matrix"', () => {
      const settings = {
        width: 1,
        height: 4,
        input: {
          width: 4,
          height: 4,
        },
        filterWidth: 4,
        filterHeight: 4
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.height);
      expect(result.length).toBe(settings.height);
      expect(result).toEqual(expected);
    });
    it('can back propagate from a "4x4x1 input matrix" and a "4x4x1 output matrix" to a "4x4x1 filter matrix"', () => {
      const settings = {
        width: 1,
        height: 4,
        input: {
          width: 4,
          height: 4
        },
        filterWidth: 4,
        filterHeight: 4
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.height);
      expect(result.length).toBe(settings.height);
      expect(result).toEqual(expected);
    });
    it('can back propagate from a "6x6x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix"', () => {
      const settings = {
        width: 1,
        height: 24,
        input: {
          width: 6,
          height: 6
        },
        filterWidth: 6,
        filterHeight: 6
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.height);
      expect(result.length).toBe(settings.height);
      expect(result).toEqual(expected);
    });
    it('can back propagate from a "12x12x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix"', () => {
      const settings = {
        width: 1,
        height: 24,
        input: {
          width: 12,
          height: 12
        },
        filterWidth: 12,
        filterHeight: 12
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.height);
      expect(result.length).toBe(settings.height);
      expect(result).toEqual(expected);
    });
    it('can back propagate from a "24x24x1 input matrix" and a "24x24x8 output matrix" to a "5x5x1 filter matrix"', () => {
      const settings = {
        width: 1,
        height: 24,
        input: {
          width: 24,
          height: 24,
        },
        filterWidth: 24,
        filterHeight: 24
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.height);
      expect(result.length).toBe(settings.height);
      expect(result).toEqual(expected);
    });
    it('can back propagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix"', () => {
      const settings = {
        width: 1,
        height: 12,
        input: {
          width: 24,
          height: 24
        },
        filterWidth: 24,
        filterHeight: 24
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.height);
      expect(result.length).toBe(settings.height);
      expect(result).toEqual(expected);
    });
    it('can back propagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix"', () => {
      const settings = {
        width: 1,
        height: 12,
        input: {
          width: 24,
          height: 24
        },
        filterWidth: 24,
        filterHeight: 24
      };

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.height);
      expect(result.length).toBe(settings.height);
      expect(result).toEqual(expected);
    });
  });
});

function write(logs) {
  const fs = require('fs');
  fs.writeFileSync('logs/deltas.log', logs.convnetMatrixLog.toString('deltas'));
  fs.writeFileSync('logs/inputs.log', logs.convnetMatrixLog.toString('inputs'));
  fs.writeFileSync('logs/deltas-new.log', logs.brainMatrixLog.toString('deltas'));
  fs.writeFileSync('logs/inputs-new.log', logs.brainMatrixLog.toString('inputs'));
}

function filterDepthHack2D(filter, settings) {
  const rows = [];
  let internalZ = 0;
  for (let y = 0; y < settings.input.height; y++) {
    const columns = [];
    for (let x = 0; x < settings.input.width; x++) {
      columns.push(filter.get_grad(0, 0, internalZ++));
    }
    rows.push(columns);
  }
  return rows;
}