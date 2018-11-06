const convnet = require('convnetjs');

function fnClassToString(className, cl) {
  return `var ${ className } = ${ cl.toString() };
  ${className}.prototype = {
    ${Object.keys(cl.prototype).map((methodName) => `${methodName}: ${ cl.prototype[methodName].toString()}`).join(',\n')}
  };`;
}

function fillZeros(width, height, depth) {
  const result = [];
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
  if (typeof depth === 'undefined') {
    if (typeof height === 'undefined') {
      for (let x = 0; x < width; x++) {
        result.push(i++);
      }
      return result;
    }
    for (let y = 0; y < height; y++) {
      const columns = [];
      for (let x = 0; x < width; x++) {
        columns.push(i++);
      }
      result.push(columns);
    }
    return result;
  }
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

function fillPlusPlusVol(width, height = 1, depth = 1) {
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

function lookupXYZ(index, width, height) {
  const x = Math.floor(index % width);
  const y = Math.floor((index / width) % height);
  const z = Math.floor(index / (width * height));
  return { x, y, z };
}

function lookupZYX(index, width, height, depth) {
  const x = Math.floor(index / (depth * height));
  const y = Math.floor((index / depth) % height);
  const z = Math.floor(index % depth);
  return { x, y, z };
}

module.exports = {
  fnClassToString,
  fillZeros,
  fillPlusPlus,
  fillPlusPlusVol,
  lookupXYZ,
  lookupZYX,
  volWToArrays,
  volDWToArrays
};