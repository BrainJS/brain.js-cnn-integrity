const convnet = require('convnetjs');
const compareFilters = require('brain.js/dist/layer/convolution').compareFilters;
const MatrixLog = require('matrix-log.js');
const gpuMock = require('gpu-mock.js');

describe('Convolution', () => {
  describe('backpropagation', () => {
    describe('algorithm shape', () => {
      function getConvNetConvLayerInstance(settings) {
        const filters = [];
        for (let i = 0; i < settings.depth; i++) {
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
          in_act: {
            sx: settings.input.width,
            sy: settings.input.height,
            depth: settings.input.depth,
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
          in_sy: settings.input.height,
          in_depth: settings.input.depth,
          sx: settings.filterWidth,
          sy: settings.filterHeight,
          out_sx: settings.width,
          out_sy: settings.height,
          out_depth: settings.depth,
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
        //const target = 'sum += deltas[this.thread.z][inputY + y][inputX + x] * inputs[this.thread.z][y][x]';
        /// start
        function compareFilters(filterDeltas, inputs, deltas) {
          const inputZ = this.thread.z;
          //
          // const deltaY = getCompareFilterDeltasIndexStart(
          //   this.thread.y,
          //   this.constants.strideY,
          //   this.constants.paddingY
          // )
          // const deltaX = getCompareFilterDeltasIndexStart(
          //   this.thread.x,
          //   this.constants.strideX,
          //   this.constants.paddingX
          // )
          // const maxDeltaY = getCompareFilterDeltasIndexStop(
          //   this.thread.y,
          //   this.constants.strideY,
          //   this.constants.paddingY,
          //   this.constants.deltasHeight,
          //   this.constants.outputHeight
          // )
          // const maxDeltaX = getCompareFilterDeltasIndexStop(
          //   this.thread.x,
          //   this.constants.strideX,
          //   this.constants.paddingX,
          //   this.constants.deltasWidth,
          //   this.constants.outputWidth
          // )

          let sum = 0//filterDeltas[this.thread.z][this.thread.y][this.thread.x];
          for (let inputY = this.thread.y - this.constants.paddingY; inputY < this.constants.inputHeight; inputY += this.constants.strideY) {
            if (inputY < 0) continue;
            for (let inputX = this.thread.x - this.constants.paddingX; inputX < this.constants.inputWidth; inputX += this.constants.strideX) {
              if (inputX < 0) continue;

              this.constants.callback({
                inputX: inputX,
                inputY: inputY,
                inputZ: inputZ,
                filterX: this.thread.x,
                filterY: this.thread.y,
                filterZ: this.thread.z
              })
            }
          }

          return sum;
        }


        /// end
        const result = compareFilters.toString()
        //   .replace(target, `\nthis.constants.callback({
        //     deltaX: inputX + x,
        //     deltaY: inputY + y,
        //     inputX: x,
        //     inputY: y,
        //     filterX: this.thread.x,
        //     filterY: this.thread.y
        //   })\n`);

        const mockInput = [
          [
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]
          ]
        ];

        return gpuMock(eval(`(${result})`), {
          output: [settings.filterWidth, settings.filterHeight, settings.input.depth],
          constants: {
            strideX: Math.max(settings.stride || 0, 1),
            strideY: Math.max(settings.stride || 0, 1),
            paddingX: Math.max(settings.padding || 0, 0),
            paddingY: Math.max(settings.padding || 0, 0),
            filterWidth: settings.filterWidth,
            filterHeight: settings.filterHeight,
            inputWidth: settings.input.width,
            inputHeight: settings.input.height,
            deltaWidth: settings.width,
            deltaHeight: settings.height,
            callback: settings.callback
          }
        })(mockInput,mockInput,mockInput);
      }
      describe('filters', () => {
        function setupLogs(settings) {
          const convnetMatrixLog = new MatrixLog('filters', settings.filterWidth, settings.filterHeight, settings.input.depth);
          const brainMatrixLog = new MatrixLog('filters', settings.filterWidth, settings.filterHeight, settings.input.depth);
          getConvNetConvLayerBackward().call(
            getConvNetConvLayerInstance(Object.assign({
              callback: (stats) => {
                if (stats.targets && stats.targets.join(',') === 'f.dw,V.dw') {

                  // if (stats.fx < 0) throw new Error('filterX less than 0');
                  // if (stats.fx > settings.filterWidth) throw new Error(`filterX greater than ${settings.filterWidth}`);
                  // if (stats.fy < 0) throw new Error('filterY less than 0');
                  // if (stats.fy > settings.filterHeight) throw new Error(`filterY greater than ${settings.filterHeight}`);
                  //
                  // if (stats.ox < 0) throw new Error('deltaX less than 0');
                  // if (stats.ox > settings.width) throw new Error(`deltaX greater than ${settings.width}`);
                  // if (stats.oy < 0) throw new Error('deltaY less than 0');
                  // if (stats.oy > settings.height) throw new Error(`deltaY greater than ${settings.height}`);
                  // if (stats.fd < 0) throw new Error('deltaZ less than 0');
                  // if (stats.fd > settings.input.depth) throw new Error(`deltaZ greater than ${settings.input.depth}`);

                  // if (stats.ax < 0) throw new Error('inputX less than 0');
                  // if (stats.ax > settings.input.width) throw new Error(`inputX greater than ${settings.input.width}`);
                  // if (stats.ay < 0) throw new Error('inputY less than 0');
                  // if (stats.ay > settings.input.height) throw new Error(`inputY greater than ${settings.input.height}`);

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
                  try {
                    convnetMatrixLog.add(inputsLog);
                  } catch (e) {
                    console.log('error in inputs', inputsLog);
                    console.log(e);
                  }
                  try {
                    convnetMatrixLog.add(deltasLog);
                  } catch (e) {
                    console.log('error in deltas', deltasLog);
                    console.log(e);
                  }
                }
              }
            }, settings)));

          getBrainConvolutionLayerCompareFilters(Object.assign({
            callback: (stats) => {
          //     // if (stats.filterX < 0) throw new Error('filterX less than 0');
          //     // if (stats.filterX > settings.filterWidth) throw new Error(`filterX greater than ${settings.filterWidth}`);
          //     // if (stats.filterY < 0) throw new Error('filterY less than 0');
          //     // if (stats.filterY > settings.filterHeight) throw new Error(`filterY greater than ${settings.filterHeight}`);
          //     //
          //     // if (stats.deltaX < 0) throw new Error('deltaX less than 0');
          //     // if (stats.deltaX > settings.width) throw new Error(`deltaX greater than ${settings.width}`);
          //     // if (stats.deltaY < 0) throw new Error('deltaY less than 0');
          //     // if (stats.deltaY > settings.height) throw new Error(`deltaY greater than ${settings.height}`);
          //     //
          //     // if (stats.inputX < 0) throw new Error('inputX less than 0');
          //     // if (stats.inputX > settings.input.width) throw new Error(`inputX greater than ${settings.input.width}`);
          //     // if (stats.inputY < 0) throw new Error('inputY less than 0');
          //     // if (stats.inputY > settings.input.height) throw new Error(`inputY greater than ${settings.input.height}`);
          //
              brainMatrixLog
                .at({
                  x: stats.filterX,
                  y: stats.filterY,
                  z: stats.filterZ
                })
                // .add({
                //   name: 'deltas',
                //   deltaX: stats.deltaX,
                //   deltaY: stats.deltaY,
                //   deltaZ: stats.deltaZ,
                //   width: stats.width,
                //   height: stats.height,
                //   depth: stats.depth
                // })
                .add({
                  name: 'inputs',
                  x: stats.inputX,
                  y: stats.inputY,
                  z: stats.inputZ,
                  width: settings.input.width,
                  height: settings.input.height,
                  depth: settings.input.depth
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
          const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
          const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
          expect(resultDeltas).toEqual(expectedDeltas);
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
          const resultDeltas = logs.brainMatrixLog.toString('deltas').split(/\n/g);
          const expectedDeltas = logs.convnetMatrixLog.toString('deltas').split(/\n/g);
          const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
          const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
          expect(resultDeltas).toEqual(expectedDeltas);
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
          const resultDeltas = logs.brainMatrixLog.toString('deltas').split(/\n/g);
          const expectedDeltas = logs.convnetMatrixLog.toString('deltas').split(/\n/g);
          const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
          const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
          expect(resultDeltas).toEqual(expectedDeltas);
          expect(resultInputs).toEqual(expectedInputs);
        });
        it.only('can backpropagate from a "12x12x8 input matrix" and a "24x24x8 output matrix" to a "5x5x8 filter matrix" with padding of 2', () => {
          const settings = {
            width: 6,
            height: 6,
            depth: 3,
            filterWidth: 6,
            filterHeight: 6,
            input: {
              width: 4,
              height: 4,
              depth: 2,
            },
            padding: 2,
            stride: 2
          };

          const logs = setupLogs(settings);
          const resultDeltas = logs.brainMatrixLog.toString('deltas').split(/\n/g);
          const expectedDeltas = logs.convnetMatrixLog.toString('deltas').split(/\n/g);
          const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
          const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
          // require('fs').writeFileSync('deltas.log', expectedDeltas.join('\n'));
          require('fs').writeFileSync('inputs.log', expectedInputs.join('\n'));
          require('fs').writeFileSync('result-inputs.log', resultInputs.join('\n'));
          // expect(resultDeltas).toEqual(expectedDeltas);
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
          const resultDeltas = logs.brainMatrixLog.toString('deltas').split(/\n/g);
          const expectedDeltas = logs.convnetMatrixLog.toString('deltas').split(/\n/g);
          const resultInputs = logs.brainMatrixLog.toString('inputs').split(/\n/g);
          const expectedInputs = logs.convnetMatrixLog.toString('inputs').split(/\n/g);
          expect(resultDeltas).toEqual(expectedDeltas);
          expect(resultInputs).toEqual(expectedInputs);
        });
      });
    });
  });
});