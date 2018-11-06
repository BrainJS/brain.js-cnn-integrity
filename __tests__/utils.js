const utils = require('../utils');

describe('utils', () => {
  describe('.lookupXYZ()', () => {
    it('looks up as expected  width=1,height=1,depth=3', () => {
      const width = 1;
      const height = 1;
      const depth = 3;

      let i = 0;
      for (let z = 0; z < depth; z++) {
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const location = utils.lookupXYZ(i++, width, height, depth);
            expect(location.x).toBe(x);
            expect(location.y).toBe(y);
            expect(location.z).toBe(z);
          }
        }
      }
    });
    it('looks up as expected  width=7,height=14,depth=3', () => {
      const width = 7;
      const height = 14;
      const depth = 3;

      let i = 0;
      for (let z = 0; z < depth; z++) {
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const location = utils.lookupXYZ(i++, width, height, depth);
            expect(location.x).toBe(x);
            expect(location.y).toBe(y);
            expect(location.z).toBe(z);
          }
        }
      }
    });
  });

  describe('.lookupZYX()', () => {
    it('looks up as expected width=1,height=1,depth=3', () => {
      const width = 1;
      const height = 1;
      const depth = 3;

      let i = 0;
      for (let x = 0; x < width; x++) {
        for (let y = 0; y < height; y++) {
          for (let z = 0; z < depth; z++) {
            const location = utils.lookupZYX(i++, width, height, depth);
            expect(location.x).toBe(x);
            expect(location.y).toBe(y);
            expect(location.z).toBe(z);
          }
        }
      }
    });
    it('looks up as expected width=7,height=14,depth=3', () => {
      const width = 7;
      const height = 14;
      const depth = 3;

      let i = 0;
      for (let x = 0; x < width; x++) {
        for (let y = 0; y < height; y++) {
          for (let z = 0; z < depth; z++) {
            const location = utils.lookupZYX(i++, width, height, depth);
            expect(location.x).toBe(x);
            expect(location.y).toBe(y);
            expect(location.z).toBe(z);
          }
        }
      }
    });
  });
});