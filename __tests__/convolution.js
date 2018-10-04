const convnet = require('convnetjs');
const compareFilters = require('brain.js/dist/layer/convolution').compareFilters;
const MatrixLog = require('matrix-log.js');
const gpuMock = require('gpu-mock.js');

const shortenResults = false;

describe('Convolution', () => {
  describe('backpropagation', () => {
    describe('filters', () => {
      function getConvNetConvLayerInstance(settings) {
        const filters = [];
        for (let i = 0; i < settings.depth; i++) {
          const filter = fillPlusPlusVol(settings.filterWidth, settings.filterHeight, settings.input.depth);
          filters.push(filter);
        }
        const outAct = fillPlusPlusVol(settings.width, settings.height, settings.depth);

        const instance = {
          in_act: fillPlusPlusVol(settings.input.width, settings.input.height, settings.input.depth),

          filters: filters,
          biases: {
            dw: new Array(settings.depth).fill(0)
          },
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
        function getBrainConvolutionLayerCompareFilters(settings) {
          const target = 'sum += input * deltas[this.constants.deltaZ][deltaY][deltaX]';
          const compareFiltersString = compareFilters.toString();
          if (compareFiltersString.indexOf(target) < 0) {
            throw new Error(`function injection target of "${target}" cannot be found`);
          }
          const compareFiltersInjectedString = compareFiltersString
            .replace(target, target + `\nthis.constants.callback({
              inputX: inputX,
              inputY: inputY,
              inputZ: this.thread.z,
              deltaX: deltaX,
              deltaY: deltaY,
              deltaZ: this.constants.deltaZ,
              filterX: this.thread.x,
              filterY: this.thread.y,
              filterZ: this.thread.z
            })\n`);
          const compareFiltersInjected = eval(`(${compareFiltersInjectedString})`);

          const filterDeltas = fillPlusPlus(settings.filterWidth, settings.filterHeight, settings.input.depth);
          const inputs = fillPlusPlus(settings.input.width, settings.input.height, settings.input.depth);
          const deltas = fillPlusPlus(settings.width, settings.height, settings.depth);

          const stride = Math.max(settings.stride || 0, 1);
          const padding = Math.max(settings.padding || 0, 0);
          const paddedInputWidth = settings.input.width + padding;
          const paddedInputHeight = settings.input.height + padding;
          const slideWidth = Math.min(settings.width, paddedInputWidth);
          const slideHeight = Math.min(settings.height, paddedInputHeight);

          const output = [];
          for (let i = 0; i < settings.depth; i++) {
            output.push(gpuMock(compareFiltersInjected, {
              output: [settings.filterWidth, settings.filterHeight, settings.input.depth],
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
                deltaZ: i,
                slideWidth: slideWidth,
                slideHeight: slideHeight
              }
            })(filterDeltas, inputs, deltas));
          }
          return output;
        }
        function setupLogs(settings) {
          const convnetMatrixLog = new MatrixLog('filters', settings.filterWidth, settings.filterHeight, settings.input.depth);
          const brainMatrixLog = new MatrixLog('filters', settings.filterWidth, settings.filterHeight, settings.input.depth);
          getConvNetConvLayerBackward().call(
            getConvNetConvLayerInstance(Object.assign({
              callback: (stats) => {
                if (stats.targets && stats.targets.join(',') === 'f.dw,V.dw') {
                  convnetMatrixLog
                    .at({
                      x: stats.fx,
                      y: stats.fy,
                      z: stats.fd
                    });

                  // in `backward` called in_act, V, or V.w
                  const inputsLog = {
                    name: 'inputs',
                    x: stats.ox,
                    y: stats.oy,
                    z: stats.fd,
                    width: settings.input.width,
                    height: settings.input.height,
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

                  convnetMatrixLog.add(inputsLog);
                  convnetMatrixLog.add(deltasLog);
                }
              }
            }, settings)));

          getBrainConvolutionLayerCompareFilters(Object.assign({
            callback: (stats) => {
              if (stats.filterX < 0) throw new Error('filterX less than 0');
              if (stats.filterX > settings.filterWidth) throw new Error(`filterX greater than ${settings.filterWidth}`);
              if (stats.filterY < 0) throw new Error('filterY less than 0');
              if (stats.filterY > settings.filterHeight) throw new Error(`filterY greater than ${settings.filterHeight}`);

              if (stats.deltaX < 0) throw new Error('deltaX less than 0');
              if (stats.deltaX > settings.width) throw new Error(`deltaX greater than ${settings.width}`);
              if (stats.deltaY < 0) throw new Error('deltaY less than 0');
              if (stats.deltaY > settings.height) throw new Error(`deltaY greater than ${settings.height}`);

              if (stats.inputX < 0) throw new Error('inputX less than 0');
              if (stats.inputX > settings.input.width) throw new Error(`inputX greater than ${settings.input.width}`);
              if (stats.inputY < 0) throw new Error('inputY less than 0');
              if (stats.inputY > settings.input.height) throw new Error(`inputY greater than ${settings.input.height}`);

              brainMatrixLog
                .at({
                  x: stats.filterX,
                  y: stats.filterY,
                  z: stats.filterZ
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
        describe('from inputs', () => {
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
            const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
            const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
            if (shortenResults) {
              resultInputs.length = 200;
              expectedInputs.length = 200;
            }
            expect(resultInputs).toEqual(expectedInputs);
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
            const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
            const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
            if (shortenResults) {
              resultInputs.length = 200;
              expectedInputs.length = 200;
            }
            expect(resultInputs).toEqual(expectedInputs);
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
            const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
            const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
            if (shortenResults) {
              resultInputs.length = 200;
              expectedInputs.length = 200;
            }
            expect(resultInputs).toEqual(expectedInputs);
          });
          it('can backpropagate from a "6x6x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix" with padding of 2 and stride of 2', () => {
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
            const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
            const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
            if (shortenResults) {
              resultInputs.length = 200;
              expectedInputs.length = 200;
            }
            expect(resultInputs).toEqual(expectedInputs);
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
            const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
            const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
            if (shortenResults) {
              resultInputs.length = 200;
              expectedInputs.length = 200;
            }
            expect(resultInputs).toEqual(expectedInputs);
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
            const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
            const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
            if (shortenResults) {
              resultInputs.length = 200;
              expectedInputs.length = 200;
            }
            expect(resultInputs).toEqual(expectedInputs);
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
            const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
            const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
            if (shortenResults) {
              resultInputs.length = 200;
              expectedInputs.length = 200;
            }
            expect(resultInputs).toEqual(expectedInputs);
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
      describe('output', () => {
        function setupCompareFilters(settings, deltaZ) {
          const stride = Math.max(settings.stride || 0, 1);
          const padding = Math.max(settings.padding || 0, 0);
          const paddedInputWidth = settings.input.width + padding;
          const paddedInputHeight = settings.input.height + padding;
          const slideWidth = Math.min(settings.width, paddedInputWidth);
          const slideHeight = Math.min(settings.height, paddedInputHeight);

          return gpuMock(compareFilters, {
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
              deltaDepth: settings.depth,
              slideWidth: slideWidth,
              slideHeight: slideHeight
            }
          });
        }
        function setupOutputs(settings) {
          const convnetInstance = getConvNetConvLayerInstance(settings);
          const filterDeltas = fillPlusPlus(settings.filterWidth, settings.filterHeight, settings.input.depth);
          const inputs = fillPlusPlus(settings.input.width, settings.input.height, settings.input.depth);
          const deltas = fillPlusPlus(settings.width, settings.height, settings.depth);

          for (let i = 0; i < settings.input.depth; i++) {
            expect(filterDeltas).toEqual(volDWToArrays(convnetInstance.filters[i]));
          }

          expect(inputs).toEqual(volWToArrays(convnetInstance.in_act));
          expect(deltas).toEqual((volDWToArrays(convnetInstance.out_act)));

          const compareFilters = [];
          for (let i = 0; i < settings.depth; i++) {
            compareFilters.push(setupCompareFilters(settings, i));
          }
          convnet.ConvLayer.prototype.backward.call(convnetInstance);
          const expected = convnetInstance.filters.map(volDWToArrays);
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
  });
});

function fillZeros(width, height, depth) {
  const result = [];
  let i = 1;
  for (let z = 0; z < depth; z++) {
    const rows = [];
    for (let y = 0; y < height; y++) {
      const columns = [];
      for (let x = 0; x < width; x++) {
        columns.push(0);
      }
      rows.push(columns);
    }
    result.push(rows);
  }
  return result;
}

function fillPlusPlus(width, height, depth) {
  const result = [];
  let i = 1;
  for (let z = 0; z < depth; z++) {
    const rows = [];
    for (let y = 0; y < height; y++) {
      const columns = [];
      for (let x = 0; x < width; x++) {
        columns.push(i++);
      }
      rows.push(columns);
    }
    result.push(rows);
  }
  return result;
}

function fillPlusPlusVol(width, height, depth) {
  const result = new convnet.Vol(width, height, depth);
  let i = 1;
  for (let z = 0; z < depth; z++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        result.set(x, y, z, i);
        result.set_grad(x, y, z, i);
        i++;
      }
    }
  }
  return result;
}

function volWToArrays(vol) {
  const result = [];
  for (let z = 0; z < vol.depth; z++) {
    const rows = []
    for (let y = 0; y < vol.sy; y++) {
      const columns = [];
      for (let x = 0; x < vol.sx; x++) {
        const value = vol.get(x, y, z);
        columns.push(value);
      }
      rows.push(columns);
    }
    result.push(rows);
  }
  return result;
}

function volDWToArrays(vol) {
  const result = [];
  for (let z = 0; z < vol.depth; z++) {
    const rows = [];
    for (let y = 0; y < vol.sy; y++) {
      const columns = [];
      for (let x = 0; x < vol.sx; x++) {
        const value = vol.get_grad(x, y, z);
        columns.push(value);
      }
      rows.push(columns);
    }
    result.push(rows);
  }
  return result;
}