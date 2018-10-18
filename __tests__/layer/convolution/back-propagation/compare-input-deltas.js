const convnet = require('convnetjs');
const compareInputDeltas = require('brain.js/dist/layer/convolution').compareInputDeltas;
const MatrixLog = require('matrix-log.js');
const gpuMock = require('gpu-mock.js');
const utils = require('../../../../utils');

const shortenResults = true;

describe('layer.Convolution.compareInputDeltas()', () => {
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
      const value = `this.callback({
        targets: ['f.dw', 'V.dw'],
        x, y, d,
        ox, oy,
        fx, fy, fd,
        ax, ay
      });`;

      const target = 'V.dw[ix1] += f.w[ix2]*chain_grad;';

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
    function getBrainConvolutionLayerCompareInputDeltas(settings) {
      const target = 'sum += filters[this.thread.z][filterY][filterX] * deltas[this.constants.deltaZ][deltaX][deltaY];';
      const compareInputDeltasString = compareInputDeltas.toString();
      if (compareInputDeltasString.indexOf(target) < 0) {
        throw new Error(`function injection target of "${target}" cannot be found`);
      }
      const compareInputDeltasInjectedString = compareInputDeltasString
        .replace(target, target + `\nthis.constants.callback({
          filterX: filterY,
          filterY: filterX,
          filterZ: 0,
          deltaX: deltaY,
          deltaY: deltaX,
          deltaZ: this.constants.deltaZ,
          inputDeltaX: this.thread.x,
          inputDeltaY: this.thread.y,
          inputDeltaZ: this.thread.z
        })\n`);
      const compareInputDeltasInjected = eval(`(${compareInputDeltasInjectedString})`);

      const inputDeltas = utils.fillPlusPlus(settings.input.width, settings.input.height, settings.input.depth);
      const filters = utils.fillPlusPlus(settings.filterWidth, settings.filterHeight, settings.input.depth);
      const deltas = utils.fillPlusPlus(settings.width, settings.height, settings.depth);

      const stride = Math.max(settings.stride || 0, 1);
      const padding = Math.max(settings.padding || 0, 0);

      const output = [];
      for (let i = 0; i < settings.depth; i++) {
        output.push(gpuMock(compareInputDeltasInjected, {
          output: [settings.input.width, settings.input.height, settings.input.depth],
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
            callback: settings.callback,
            deltaZ: i
          }
        })(inputDeltas, filters, deltas));
      }
      return output;
    }
    function setupLogs(settings) {
      const convnetMatrixLog = new MatrixLog('inputDeltas', settings.input.width, settings.input.height, settings.input.depth);
      const brainMatrixLog = new MatrixLog('inputDeltas', settings.input.width, settings.input.height, settings.input.depth);
      getConvNetConvLayerBackward().call(
        getConvNetConvLayerInstance(Object.assign({
          callback: (stats) => {
            if (stats.targets && stats.targets.join(',') === 'f.dw,V.dw') {

              convnetMatrixLog
                .at({
                  x: stats.oy,
                  y: stats.ox,
                  z: stats.fd
                });

              // in `backward` called in_act, V, or V.w
              const filtersLog = {
                name: 'filters',
                x: stats.fx,
                y: stats.fy,
                z: stats.fd,
                width: settings.filterWidth,
                height: settings.filterHeight,
                depth: settings.input.depth
              };

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

              convnetMatrixLog.add(filtersLog);
              convnetMatrixLog.add(deltasLog);
            }
          }
        }, settings)));

      getBrainConvolutionLayerCompareInputDeltas(Object.assign({
        callback: (stats) => {
          // if (stats.filterX < 0) throw new Error('filterX less than 0');
          // if (stats.filterX > settings.filterWidth) throw new Error(`filterX greater than ${settings.filterWidth}`);
          // if (stats.filterY < 0) throw new Error('filterY less than 0');
          // if (stats.filterY > settings.filterHeight) throw new Error(`filterY greater than ${settings.filterHeight}`);
          //
          // if (stats.deltaX < 0) throw new Error('deltaX less than 0');
          // if (stats.deltaX > settings.width) throw new Error(`deltaX greater than ${settings.width}`);
          // if (stats.deltaY < 0) throw new Error('deltaY less than 0');
          // if (stats.deltaY > settings.height) throw new Error(`deltaY greater than ${settings.height}`);
          //
          // if (stats.inputX < 0) throw new Error('inputX less than 0');
          // if (stats.inputX > settings.input.width) throw new Error(`inputX greater than ${settings.input.width}`);
          // if (stats.inputY < 0) throw new Error('inputY less than 0');
          // if (stats.inputY > settings.input.height) throw new Error(`inputY greater than ${settings.input.height}`);

          brainMatrixLog
            .at({
              x: stats.inputDeltaX,
              y: stats.inputDeltaY,
              z: stats.inputDeltaZ
            });
          brainMatrixLog
            .add({
              name: 'filters',
              x: stats.filterX,
              y: stats.filterY,
              z: stats.filterZ,
              width: settings.filterWidth,
              height: settings.filterHeight,
              depth: settings.input.depth
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
    describe('from filters', () => {
      it('can backpropagate from a "4x4x1 input matrix" and a "1x1x1 output matrix" to a "2x2x1 filter matrix"', () => {
        const settings = {
          width: 3,
          height: 3,
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
        write(logs);
        const resultFilters = logs.brainMatrixLog.toString('filters').split(/\n/g);
        const expectedFilters = logs.convnetMatrixLog.toString('filters').split(/\n/g);
        if (shortenResults) {
          resultFilters.length = 200;
          expectedFilters.length = 200;
        }
        expect(resultFilters).toEqual(expectedFilters);
      });
      it('can backpropagate from a "4x4x1 input matrix" and a "2x2x1 output matrix" to a "2x2x1 filter matrix"', () => {
        const settings = {
          width: 12,
          height: 12,
          depth: 1,
          filterWidth: 6,
          filterHeight: 6,
          input: {
            width: 14,
            height: 14,
            depth: 1,
          },
          stride: 2,
          padding: 2
        };

        const logs = setupLogs(settings);
        const resultFilters = logs.brainMatrixLog.toString('filters').split(/\n/g);
        const expectedFilters = logs.convnetMatrixLog.toString('filters').split(/\n/g);
        if (shortenResults) {
          resultFilters.length = 200;
          expectedFilters.length = 200;
        }
        expect(resultFilters).toEqual(expectedFilters);
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
        const resultFilters = logs.brainMatrixLog.toString('filters').split(/\n/g);
        const expectedFilters = logs.convnetMatrixLog.toString('filters').split(/\n/g);
        if (shortenResults) {
          resultFilters.length = 200;
          expectedFilters.length = 200;
        }
        expect(resultFilters).toEqual(expectedFilters);
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
        const resultFilters = logs.brainMatrixLog.toString('filters').split(/\n/g);
        const expectedFilters = logs.convnetMatrixLog.toString('filters').split(/\n/g);
        if (shortenResults) {
          resultFilters.length = 200;
          expectedFilters.length = 200;
        }
        expect(resultFilters).toEqual(expectedFilters);
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
          stride: 1
        };

        const logs = setupLogs(settings);
        const resultFilters = logs.brainMatrixLog.toString('filters').split(/\n/g);
        const expectedFilters = logs.convnetMatrixLog.toString('filters').split(/\n/g);
        if (shortenResults) {
          resultFilters.length = 200;
          expectedFilters.length = 200;
        }
        expect(resultFilters).toEqual(expectedFilters);
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
        const resultFilters = logs.brainMatrixLog.toString('filters').split(/\n/g);
        const expectedFilters = logs.convnetMatrixLog.toString('filters').split(/\n/g);
        if (shortenResults) {
          resultFilters.length = 200;
          expectedFilters.length = 200;
        }
        expect(resultFilters).toEqual(expectedFilters);
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
        const resultFilters = logs.brainMatrixLog.toString('filters').split(/\n/g);
        const expectedFilters = logs.convnetMatrixLog.toString('filters').split(/\n/g);
        if (shortenResults) {
          resultFilters.length = 200;
          expectedFilters.length = 200;
        }
        expect(resultFilters).toEqual(expectedFilters);
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
        const resultFilters = logs.brainMatrixLog.toString('filters').split(/\n/g);
        const expectedFilters = logs.convnetMatrixLog.toString('filters').split(/\n/g);
        if (shortenResults) {
          resultFilters.length = 200;
          expectedFilters.length = 200;
        }
        expect(resultFilters).toEqual(expectedFilters);
      });
    });
    describe('from deltas', () => {
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
          width: 12,
          height: 12,
          depth: 1,
          filterWidth: 6,
          filterHeight: 6,
          input: {
            width: 14,
            height: 14,
            depth: 1,
          },
          stride: 2,
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

        // require('fs').writeFileSync('logs/deltas.log', logs.convnetMatrixLog.toString('deltas'));
        // require('fs').writeFileSync('logs/filters.log', logs.convnetMatrixLog.toString('filters'));
        // require('fs').writeFileSync('logs/deltas-new.log', logs.brainMatrixLog.toString('deltas'));

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
  });
  describe.skip('output', () => {
    function setupCompareInputDeltas(settings, deltaZ) {
      const stride = Math.max(settings.stride || 0, 1);
      const padding = Math.max(settings.padding || 0, 0);

      return gpuMock(compareInputDeltas, {
        output: [settings.filterWidth, settings.filterHeight, settings.input.depth],
        constants: {
          deltaZ: deltaZ,
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
          deltaDepth: settings.depth
        }
      });
    }
    function setupOutputs(settings) {
      const convnetInstance = getConvNetConvLayerInstance(settings);
      const filterDeltas = utils.fillPlusPlus(settings.filterWidth, settings.filterHeight, settings.input.depth);
      const inputs = utils.fillPlusPlus(settings.input.width, settings.input.height, settings.input.depth);
      const deltas = utils.fillPlusPlus(settings.width, settings.height, settings.depth);

      for (let i = 0; i < settings.input.depth; i++) {
        expect(filterDeltas).toEqual(utils.volDWToArrays(convnetInstance.filters[i]));
      }

      expect(inputs).toEqual(utils.volWToArrays(convnetInstance.in_act));
      expect(deltas).toEqual((utils.volDWToArrays(convnetInstance.out_act)));

      const compareFilters = [];
      for (let i = 0; i < settings.depth; i++) {
        compareFilters.push(setupCompareInputDeltas(settings, i));
      }
      convnet.ConvLayer.prototype.backward.call(convnetInstance);
      const expected = convnetInstance.filters.map(utils.volDWToArrays);
      const result = [];
      for (let i = 0; i < settings.depth; i++) {
        result.push(compareFilters[i](filterDeltas, inputs, deltas));
      }
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

function write(logs) {
  const fs = require('fs');
  fs.writeFileSync('logs/deltas.log', logs.convnetMatrixLog.toString('deltas'));
  fs.writeFileSync('logs/filters.log', logs.convnetMatrixLog.toString('filters'));
  fs.writeFileSync('logs/deltas-new.log', logs.brainMatrixLog.toString('deltas'));
  fs.writeFileSync('logs/filters-new.log', logs.brainMatrixLog.toString('filters'));
}