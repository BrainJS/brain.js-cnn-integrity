const convnet = require('convnetjs');
const predict = require('brain.js/dist/layer/convolution').predict;
const MatrixLog = require('matrix-log.js');
const gpuMock = require('gpu-mock.js');
const utils = require('../../../../utils');

const shortenResults = false;

describe('layer.Convolution.predict()', () => {
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
      filtersAndInputsCallback: settings.filtersAndInputsCallback,
      biasesCallback: settings.biasesCallback,
    };
    return instance;
  }
  describe('algorithm shape', () => {
    function getConvNetConvLayerForward() {
      const value1 = `this.filtersAndInputsCallback({
        x, y, d,
        ox, oy,
        fx, fy, fd,
        ax, ay
      });`;
      const value2 = `this.biasesCallback({
        d,
        ax, ay
      });`;

      const target1 = 'a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V.sx * oy)+ox)*V.depth+fd];';
      const target2 = 'a += this.biases.w[d];';
      const forwardString = convnet.ConvLayer.prototype.forward.toString();

      if (forwardString.indexOf(target1) < 0) {
        throw new Error(`function injection target of "${target1}" cannot be found`);
      }
      if (forwardString.indexOf(target2) < 0) {
        throw new Error(`function injection target of "${target2}" cannot be found`);
      }

      const result = forwardString
        .replace('{', `{\n${ utils.fnClassToString('Vol', convnet.Vol) }\n`)
        .replace(target1,
          target1 + '\n' + value1 + '\n')
        .replace(target2,
          target2 + '\n' + value2 + '\n')
        .replace(/global\.zeros/g, 'new Array');

      return eval(`(${result})`);
    }
    function getBrainConvolutionLayerPredict(settings) {
      const target1 = 'sum += filters[z][filterY][filterX] * inputs[z][inputY][inputX];';
      const target2 = 'return sum + biases[this.thread.z];';
      const predictString = predict.toString();
      if (predictString.indexOf(target1) < 0) {
        throw new Error(`function injection target of "${target1}" cannot be found`);
      }
      if (predictString.indexOf(target2) < 0) {
        throw new Error(`function injection target of "${target2}" cannot be found`);
      }
      const predictInjectedString = predictString
        .replace(target1, target1 + `\nthis.constants.filtersAndInputsCallback({
          inputX: inputX,
          inputY: inputY,
          inputZ: z,
         
          filterX: filterX,
          filterY: filterY,
          filterZ: z,
          
          x: this.thread.x,
          y: this.thread.y,
          z: this.thread.z
        })\n`)
        .replace(target2, `\nthis.constants.biasesCallback({
          x: this.thread.x,
          y: this.thread.y,
          z: this.thread.z
        })\n` + target2);
      const predictInjected = eval(`(${predictInjectedString})`);

      const filters = utils.fillPlusPlus(settings.filterWidth, settings.filterHeight, settings.input.depth);
      const inputs = utils.fillPlusPlus(settings.input.width, settings.input.height, settings.input.depth);
      const biases = utils.fillPlusPlus(settings.input.depth);

      const stride = Math.max(settings.stride || 0, 1);
      const padding = Math.max(settings.padding || 0, 0);

      return gpuMock(predictInjected, {
        output: [settings.width, settings.height, settings.depth],
        constants: {
          strideX: stride,
          strideY: stride,
          paddingX: padding,
          paddingY: padding,
          filterWidth: settings.filterWidth,
          filterHeight: settings.filterHeight,
          filterCount: settings.input.depth,
          inputWidth: settings.input.width,
          inputHeight: settings.input.height,
          inputDepth: settings.input.depth,
          deltaWidth: settings.width,
          deltaHeight: settings.height,
          deltaDepth: settings.depth,
          filtersAndInputsCallback: settings.filtersAndInputsCallback,
          biasesCallback: settings.biasesCallback
        }
      })(inputs, filters, biases);
    }
    function setupLogs(settings) {
      const convnetMatrixLog = new MatrixLog('weights', settings.width, settings.height, settings.depth);
      const brainMatrixLog = new MatrixLog('weights', settings.width, settings.height, settings.depth);
      getConvNetConvLayerForward().call(
        getConvNetConvLayerInstance(Object.assign({
          filtersAndInputsCallback: (stats) => {
            convnetMatrixLog
              .at({
                x: stats.ax,
                y: stats.ay,
                z: stats.d
              });

            // in `forward` called in_act, V, or V.w
            const inputsLog = {
              name: 'inputs',
              x: stats.ox,
              y: stats.oy,
              z: stats.fd,
              width: settings.input.width,
              height: settings.input.height,
              depth: settings.input.depth
            };

            const filtersLog = {
              name: 'filters',
              x: stats.fx,
              y: stats.fy,
              z: stats.fd,
              width: settings.filterWidth,
              height: settings.filterHeight,
              depth: settings.input.depth
            };

            convnetMatrixLog.add(inputsLog);
            convnetMatrixLog.add(filtersLog);
          },
          biasesCallback: (stats) => {
            convnetMatrixLog
              .at({
                x: stats.ax,
                y: stats.ay,
                z: stats.d
              });
            const biasLog = {
              name: 'biases',
              x: stats.d,
              y: 0,
              width: settings.input.depth,
              height: 1
            };
            convnetMatrixLog.add(biasLog);
          }
        }, settings)), utils.fillPlusPlusVol(settings.input.width, settings.input.height, settings.input.depth));

      getBrainConvolutionLayerPredict(Object.assign({
        filtersAndInputsCallback: (stats) => {
          if (stats.filterX < 0) throw new Error('filterX less than 0');
          if (stats.filterX > settings.filterWidth) throw new Error(`filterX greater than ${settings.filterWidth}`);
          if (stats.filterY < 0) throw new Error('filterY less than 0');
          if (stats.filterY > settings.filterHeight) throw new Error(`filterY greater than ${settings.filterHeight}`);

          if (stats.inputX < 0) throw new Error('inputX less than 0');
          if (stats.inputX > settings.input.width) throw new Error(`inputX greater than ${settings.input.width}`);
          if (stats.inputY < 0) throw new Error('inputY less than 0');
          if (stats.inputY > settings.input.height) throw new Error(`inputY greater than ${settings.input.height}`);

          brainMatrixLog
            .at({
              x: stats.x,
              y: stats.y,
              z: stats.z
            });
          brainMatrixLog
            .add({
              name: 'inputs',
              x: stats.inputX,
              y: stats.inputY,
              z: stats.inputZ,
              width: settings.input.width,
              height: settings.input.height,
              depth: settings.input.depth
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
        },
        biasesCallback: (stats) => {
          brainMatrixLog
            .at({
              x: stats.x,
              y: stats.y,
              z: stats.z
            });
          brainMatrixLog
            .add({
              name: 'biases',
              x: stats.z,
              y: 0,
              width: settings.input.depth,
              height: 1,
            });
        }
      }, settings));

      return { convnetMatrixLog, brainMatrixLog };
    }
    describe('from filters', () => {
      it('can forward propagate from a "4x4x1 input matrix" and a "3x3x1 output matrix" to a "2x2x1 filter matrix"', () => {
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
        const resultFilters = logs.brainMatrixLog.toString('filters').split(/\n/g);
        const expectedFilters = logs.convnetMatrixLog.toString('filters').split(/\n/g);
        if (shortenResults) {
          resultFilters.length = 200;
          expectedFilters.length = 200;
        }
        expect(resultFilters).toEqual(expectedFilters);
      });

      it('can forward propagate from a "4x4x1 input matrix" and a "3x3x1 output matrix" to a "2x2x1 filter matrix"', () => {
        // output should always be the input size + (padding * 2) - filter size + 1
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
        const resultFilters = logs.brainMatrixLog.toString('filters').split(/\n/g);
        const expectedFilters = logs.convnetMatrixLog.toString('filters').split(/\n/g);
        if (shortenResults) {
          resultFilters.length = 200;
          expectedFilters.length = 200;
        }
        expect(resultFilters).toEqual(expectedFilters);
      });
      it('can forward propagate from a "4x4x1 input matrix" and a "4x4x1 output matrix" to a "4x4x1 filter matrix"', () => {
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
      it('can forward propagate from a "6x6x8 input matrix" and a "24x24x8 output matrix" to a "10x10x8 filter matrix" with padding of 2 and stride of 2', () => {
        const settings = {
          width: 24,
          height: 24,
          depth: 8,
          filterWidth: 10,
          filterHeight: 10,
          input: {
            width: 6,
            height: 6,
            depth: 5,
          },
          padding: 2,
          stride: 2
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
      it('can forward propagate from a "12x12x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix" with padding of 2', () => {
        const settings = {
          width: 6,
          height: 6,
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
        const resultFilters = logs.brainMatrixLog.toString('filters').split(/\n/g);
        const expectedFilters = logs.convnetMatrixLog.toString('filters').split(/\n/g);
        if (shortenResults) {
          resultFilters.length = 200;
          expectedFilters.length = 200;
        }
        expect(resultFilters).toEqual(expectedFilters);
      });
      it('can forward propagate from a "24x24x1 input matrix" and a "24x24x8 output matrix" to a "5x5x1 filter matrix" with padding of 2', () => {
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
      it('can forward propagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix" with padding of 2', () => {
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
      it('can forward propagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix" with padding of 2 and stride of 2', () => {
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
    describe('from inputs', () => {
      it('can forward propagate from a "4x4x1 input matrix" and a "3x3x1 output matrix" to a "2x2x1 filter matrix"', () => {
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
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can forward propagate from a "4x4x1 input matrix" and a "2x2x1 output matrix" to a "2x2x1 filter matrix"', () => {
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
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can forward propagate from a "4x4x1 input matrix" and a "4x4x1 output matrix" to a "4x4x1 filter matrix"', () => {
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
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can forward propagate from a "6x6x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix" with padding of 2 and stride of 2', () => {
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
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can forward propagate from a "12x12x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix" with padding of 2', () => {
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
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can forward propagate from a "24x24x1 input matrix" and a "24x24x8 output matrix" to a "5x5x1 filter matrix" with padding of 2', () => {
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
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can forward propagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix" with padding of 2', () => {
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
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
      it('can forward propagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix" with padding of 2 and stride of 2', () => {
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
        const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
        const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
        if (shortenResults) {
          resultInputs.length = 200;
          expectedInputs.length = 200;
        }
        expect(resultInputs).toEqual(expectedInputs);
      });
    });
    describe('from biases', () => {
      it('can forward propagate from a "4x4x1 input matrix" and a "3x3x1 output matrix" to a "2x2x1 filter matrix"', () => {
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
        const resultBiases = logs.brainMatrixLog.toString('biases').split(/\n/g);
        const expectedBiases = logs.convnetMatrixLog.toString('biases').split(/\n/g);
        if (shortenResults) {
          resultBiases.length = 200;
          expectedBiases.length = 200;
        }
        expect(resultBiases).toEqual(expectedBiases);
      });
      it('can forward propagate from a "4x4x1 input matrix" and a "2x2x1 output matrix" to a "2x2x1 filter matrix"', () => {
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
        const resultBiases = logs.brainMatrixLog.toString('biases').split(/\n/g);
        const expectedBiases = logs.convnetMatrixLog.toString('biases').split(/\n/g);
        if (shortenResults) {
          resultBiases.length = 200;
          expectedBiases.length = 200;
        }
        expect(resultBiases).toEqual(expectedBiases);
      });
      it('can forward propagate from a "4x4x1 input matrix" and a "4x4x1 output matrix" to a "4x4x1 filter matrix"', () => {
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
        const resultBiases = logs.brainMatrixLog.toString('biases').split(/\n/g);
        const expectedBiases = logs.convnetMatrixLog.toString('biases').split(/\n/g);
        if (shortenResults) {
          resultBiases.length = 200;
          expectedBiases.length = 200;
        }
        expect(resultBiases).toEqual(expectedBiases);
      });
      it('can forward propagate from a "6x6x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix" with padding of 2 and stride of 2', () => {
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
        const resultBiases = logs.brainMatrixLog.toString('biases').split(/\n/g);
        const expectedBiases = logs.convnetMatrixLog.toString('biases').split(/\n/g);
        if (shortenResults) {
          resultBiases.length = 200;
          expectedBiases.length = 200;
        }
        expect(resultBiases).toEqual(expectedBiases);
      });
      it('can forward propagate from a "12x12x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix" with padding of 2', () => {
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
        const resultBiases = logs.brainMatrixLog.toString('biases').split(/\n/g);
        const expectedBiases = logs.convnetMatrixLog.toString('biases').split(/\n/g);
        if (shortenResults) {
          resultBiases.length = 200;
          expectedBiases.length = 200;
        }
        expect(resultBiases).toEqual(expectedBiases);
      });
      it('can forward propagate from a "24x24x1 input matrix" and a "24x24x8 output matrix" to a "5x5x1 filter matrix" with padding of 2', () => {
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
        const resultBiases = logs.brainMatrixLog.toString('biases').split(/\n/g);
        const expectedBiases = logs.convnetMatrixLog.toString('biases').split(/\n/g);
        if (shortenResults) {
          resultBiases.length = 200;
          expectedBiases.length = 200;
        }
        expect(resultBiases).toEqual(expectedBiases);
      });
      it('can forward propagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix" with padding of 2', () => {
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
        const resultBiases = logs.brainMatrixLog.toString('biases').split(/\n/g);
        const expectedBiases = logs.convnetMatrixLog.toString('biases').split(/\n/g);
        if (shortenResults) {
          resultBiases.length = 200;
          expectedBiases.length = 200;
        }
        expect(resultBiases).toEqual(expectedBiases);
      });
      it('can forward propagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix" with padding of 2 and stride of 2', () => {
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
        const resultBiases = logs.brainMatrixLog.toString('biases').split(/\n/g);
        const expectedBiases = logs.convnetMatrixLog.toString('biases').split(/\n/g);
        if (shortenResults) {
          resultBiases.length = 200;
          expectedBiases.length = 200;
        }
        expect(resultBiases).toEqual(expectedBiases);
      });
    });
  });
  describe('output', () => {
    function setupPredict(settings) {
      const stride = Math.max(settings.stride || 0, 1);
      const padding = Math.max(settings.padding || 0, 0);

      return gpuMock(predict, {
        output: [settings.width, settings.height, settings.depth],
        constants: {
          strideX: stride,
          strideY: stride,
          paddingX: padding,
          paddingY: padding,
          filterWidth: settings.filterWidth,
          filterHeight: settings.filterHeight,
          inputWidth: settings.input.width,
          inputHeight: settings.input.height,
          inputDepth: settings.input.depth,
          deltaWidth: settings.width,
          deltaHeight: settings.height,
          deltaDepth: settings.depth
        }
      });
    }
    function setupOutputs(settings) {
      const convnetInstance = getConvNetConvLayerInstance(settings);
      const filters = utils.fillPlusPlus(settings.filterWidth, settings.filterHeight, settings.input.depth);
      const inputs = utils.fillPlusPlus(settings.input.width, settings.input.height, settings.input.depth);
      const biases = utils.fillPlusPlus(settings.depth);

      for (let i = 0; i < settings.input.depth; i++) {
        expect(filters).toEqual(utils.volWToArrays(convnetInstance.filters[i]));
      }

      expect(inputs).toEqual(utils.volWToArrays(convnetInstance.in_act));

      const predict = setupPredict(settings);
      convnet.ConvLayer.prototype.forward.call(convnetInstance, utils.fillPlusPlusVol(settings.input.width, settings.input.height, settings.input.depth));
      const expected = utils.volWToArrays(convnetInstance.out_act);
      const result = predict(inputs, filters, biases);
      return {
        expected, result
      }
    }
    it('can forward propagate from a "4x4x1 input matrix" and a "1x1x1 output matrix" to a "2x2x1 filter matrix"', () => {
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

      const { result, expected } = setupOutputs(settings);

      expect(expected.length).toBe(settings.depth);
      expect(result.length).toBe(settings.depth);
      expect(result).toEqual(expected);
    });
    it('can forward propagate from a "4x4x1 input matrix" and a "2x2x1 output matrix" to a "2x2x1 filter matrix"', () => {
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
    it('can forward propagate from a "4x4x1 input matrix" and a "4x4x1 output matrix" to a "4x4x1 filter matrix"', () => {
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
    it('can forward propagate from a "6x6x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix" with padding of 2 and stride of 2', () => {
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
    it('can forward propagate from a "12x12x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix" with padding of 2', () => {
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
    it('can forward propagate from a "24x24x1 input matrix" and a "24x24x8 output matrix" to a "5x5x1 filter matrix" with padding of 2', () => {
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
    it('can forward propagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix" with padding of 2', () => {
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
    it('can forward propagate from a "24x24x1 input matrix" and a "12x12x8 output matrix" to a "5x5x1 filter matrix" with padding of 2 and stride of 2', () => {
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
  fs.writeFileSync('logs/inputs.log', logs.convnetMatrixLog.toString('inputs'));
  fs.writeFileSync('logs/filters.log', logs.convnetMatrixLog.toString('filters'));
  fs.writeFileSync('logs/inputs-new.log', logs.brainMatrixLog.toString('inputs'));
  fs.writeFileSync('logs/filters-new.log', logs.brainMatrixLog.toString('filters'));
}