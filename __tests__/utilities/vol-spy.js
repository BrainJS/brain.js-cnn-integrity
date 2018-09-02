const VolSpy = require('../../utilities/vol-spy');

describe('VolSpy', () => {
  describe('indexFromPoint', () => {
    it('can perform simple lookup of point on 2x2x2 cube', () => {
      expect(VolSpy.indexFromPoint(2, 2, 2, 0, 0, 0)).toBe(0);
      expect(VolSpy.indexFromPoint(2, 2, 2, 0, 0, 1)).toBe(1);
      expect(VolSpy.indexFromPoint(2, 2, 2, 1, 0, 0)).toBe(2);
      expect(VolSpy.indexFromPoint(2, 2, 2, 1, 0, 1)).toBe(3);
      expect(VolSpy.indexFromPoint(2, 2, 2, 0, 1, 0)).toBe(4);
      expect(VolSpy.indexFromPoint(2, 2, 2, 0, 1, 1)).toBe(5);
      expect(VolSpy.indexFromPoint(2, 2, 2, 1, 1, 0)).toBe(6);
      expect(VolSpy.indexFromPoint(2, 2, 2, 1, 1, 1)).toBe(7);
    });
    it('can perform simple lookup of point on 6x6x2 cube', () => {
      expect(VolSpy.indexFromPoint(6, 6, 2, 0, 0, 0)).toBe(0);
      expect(VolSpy.indexFromPoint(6, 6, 2, 1, 0, 0)).toBe(2);
      expect(VolSpy.indexFromPoint(6, 6, 2, 2, 0, 0)).toBe(4);
      expect(VolSpy.indexFromPoint(6, 6, 2, 3, 0, 0)).toBe(6);
      expect(VolSpy.indexFromPoint(6, 6, 2, 4, 0, 0)).toBe(8);
      expect(VolSpy.indexFromPoint(6, 6, 2, 5, 0, 0)).toBe(10);

      expect(VolSpy.indexFromPoint(6, 6, 2, 0, 1, 0)).toBe(12);
      expect(VolSpy.indexFromPoint(6, 6, 2, 1, 1, 0)).toBe(14);
      expect(VolSpy.indexFromPoint(6, 6, 2, 2, 1, 0)).toBe(16);
      expect(VolSpy.indexFromPoint(6, 6, 2, 3, 1, 0)).toBe(18);
      expect(VolSpy.indexFromPoint(6, 6, 2, 4, 1, 0)).toBe(20);
      expect(VolSpy.indexFromPoint(6, 6, 2, 5, 1, 0)).toBe(22);

      expect(VolSpy.indexFromPoint(6, 6, 2, 0, 2, 0)).toBe(24);
      expect(VolSpy.indexFromPoint(6, 6, 2, 1, 2, 0)).toBe(26);
      expect(VolSpy.indexFromPoint(6, 6, 2, 2, 2, 0)).toBe(28);
      expect(VolSpy.indexFromPoint(6, 6, 2, 3, 2, 0)).toBe(30);
      expect(VolSpy.indexFromPoint(6, 6, 2, 4, 2, 0)).toBe(32);
      expect(VolSpy.indexFromPoint(6, 6, 2, 5, 2, 0)).toBe(34);

      expect(VolSpy.indexFromPoint(6, 6, 2, 0, 3, 0)).toBe(36);
      expect(VolSpy.indexFromPoint(6, 6, 2, 1, 3, 0)).toBe(38);
      expect(VolSpy.indexFromPoint(6, 6, 2, 2, 3, 0)).toBe(40);
      expect(VolSpy.indexFromPoint(6, 6, 2, 3, 3, 0)).toBe(42);
      expect(VolSpy.indexFromPoint(6, 6, 2, 4, 3, 0)).toBe(44);
      expect(VolSpy.indexFromPoint(6, 6, 2, 5, 3, 0)).toBe(46);

      expect(VolSpy.indexFromPoint(6, 6, 2, 0, 4, 0)).toBe(48);
      expect(VolSpy.indexFromPoint(6, 6, 2, 1, 4, 0)).toBe(50);
      expect(VolSpy.indexFromPoint(6, 6, 2, 2, 4, 0)).toBe(52);
      expect(VolSpy.indexFromPoint(6, 6, 2, 3, 4, 0)).toBe(54);
      expect(VolSpy.indexFromPoint(6, 6, 2, 4, 4, 0)).toBe(56);
      expect(VolSpy.indexFromPoint(6, 6, 2, 5, 4, 0)).toBe(58);

      expect(VolSpy.indexFromPoint(6, 6, 2, 0, 5, 0)).toBe(60);
      expect(VolSpy.indexFromPoint(6, 6, 2, 1, 5, 0)).toBe(62);
      expect(VolSpy.indexFromPoint(6, 6, 2, 2, 5, 0)).toBe(64);
      expect(VolSpy.indexFromPoint(6, 6, 2, 3, 5, 0)).toBe(66);
      expect(VolSpy.indexFromPoint(6, 6, 2, 4, 5, 0)).toBe(68);
      expect(VolSpy.indexFromPoint(6, 6, 2, 5, 5, 0)).toBe(70);

      expect(VolSpy.indexFromPoint(6, 6, 2, 0, 0, 1)).toBe(1);
      expect(VolSpy.indexFromPoint(6, 6, 2, 1, 0, 1)).toBe(3);
      expect(VolSpy.indexFromPoint(6, 6, 2, 2, 0, 1)).toBe(5);
      expect(VolSpy.indexFromPoint(6, 6, 2, 3, 0, 1)).toBe(7);
      expect(VolSpy.indexFromPoint(6, 6, 2, 4, 0, 1)).toBe(9);
      expect(VolSpy.indexFromPoint(6, 6, 2, 5, 0, 1)).toBe(11);

      expect(VolSpy.indexFromPoint(6, 6, 2, 0, 1, 1)).toBe(13);
      expect(VolSpy.indexFromPoint(6, 6, 2, 1, 1, 1)).toBe(15);
      expect(VolSpy.indexFromPoint(6, 6, 2, 2, 1, 1)).toBe(17);
      expect(VolSpy.indexFromPoint(6, 6, 2, 3, 1, 1)).toBe(19);
      expect(VolSpy.indexFromPoint(6, 6, 2, 4, 1, 1)).toBe(21);
      expect(VolSpy.indexFromPoint(6, 6, 2, 5, 1, 1)).toBe(23);

      expect(VolSpy.indexFromPoint(6, 6, 2, 0, 2, 1)).toBe(25);
      expect(VolSpy.indexFromPoint(6, 6, 2, 1, 2, 1)).toBe(27);
      expect(VolSpy.indexFromPoint(6, 6, 2, 2, 2, 1)).toBe(29);
      expect(VolSpy.indexFromPoint(6, 6, 2, 3, 2, 1)).toBe(31);
      expect(VolSpy.indexFromPoint(6, 6, 2, 4, 2, 1)).toBe(33);
      expect(VolSpy.indexFromPoint(6, 6, 2, 5, 2, 1)).toBe(35);

      expect(VolSpy.indexFromPoint(6, 6, 2, 0, 3, 1)).toBe(37);
      expect(VolSpy.indexFromPoint(6, 6, 2, 1, 3, 1)).toBe(39);
      expect(VolSpy.indexFromPoint(6, 6, 2, 2, 3, 1)).toBe(41);
      expect(VolSpy.indexFromPoint(6, 6, 2, 3, 3, 1)).toBe(43);
      expect(VolSpy.indexFromPoint(6, 6, 2, 4, 3, 1)).toBe(45);
      expect(VolSpy.indexFromPoint(6, 6, 2, 5, 3, 1)).toBe(47);

      expect(VolSpy.indexFromPoint(6, 6, 2, 0, 4, 1)).toBe(49);
      expect(VolSpy.indexFromPoint(6, 6, 2, 1, 4, 1)).toBe(51);
      expect(VolSpy.indexFromPoint(6, 6, 2, 2, 4, 1)).toBe(53);
      expect(VolSpy.indexFromPoint(6, 6, 2, 3, 4, 1)).toBe(55);
      expect(VolSpy.indexFromPoint(6, 6, 2, 4, 4, 1)).toBe(57);
      expect(VolSpy.indexFromPoint(6, 6, 2, 5, 4, 1)).toBe(59);

      expect(VolSpy.indexFromPoint(6, 6, 2, 0, 5, 1)).toBe(61);
      expect(VolSpy.indexFromPoint(6, 6, 2, 1, 5, 1)).toBe(63);
      expect(VolSpy.indexFromPoint(6, 6, 2, 2, 5, 1)).toBe(65);
      expect(VolSpy.indexFromPoint(6, 6, 2, 3, 5, 1)).toBe(67);
      expect(VolSpy.indexFromPoint(6, 6, 2, 4, 5, 1)).toBe(69);
      expect(VolSpy.indexFromPoint(6, 6, 2, 5, 5, 1)).toBe(71);
    });
  });
  describe('pointFromIndex', () => {
    it('can perform simple lookup of point on 2x2x2 cube', () => {
      function xyz(x, y, z) {
        return VolSpy.pointFromIndex(2, 2, 2, VolSpy.indexFromPoint(2, 2, 2, x, y, z));
      }
      expect(xyz(0, 0, 0)).toMatchObject({ x: 0, y: 0, z: 0 });
      expect(xyz(0, 0, 1)).toMatchObject({ x: 0, y: 0, z: 1 });
      expect(xyz(1, 0, 0)).toMatchObject({ x: 1, y: 0, z: 0 });
      expect(xyz(1, 0, 1)).toMatchObject({ x: 1, y: 0, z: 1 });
      expect(xyz(0, 1, 0)).toMatchObject({ x: 0, y: 1, z: 0 });
      expect(xyz(0, 1, 1)).toMatchObject({ x: 0, y: 1, z: 1 });
      expect(xyz(1, 1, 0)).toMatchObject({ x: 1, y: 1, z: 0 });
      expect(xyz(1, 1, 1)).toMatchObject({ x: 1, y: 1, z: 1 });
    });
    it('can perform simple lookup of point on 6x6x2 cube', () => {
      function xyz(x, y, z) {
        return VolSpy.pointFromIndex(6, 6, 2, VolSpy.indexFromPoint(6, 6, 2, x, y, z));
      }
      expect(xyz(0, 0, 0)).toMatchObject({ x: 0, y: 0, z: 0 });
      expect(xyz(1, 0, 0)).toMatchObject({ x: 1, y: 0, z: 0 });
      expect(xyz(2, 0, 0)).toMatchObject({ x: 2, y: 0, z: 0 });
      expect(xyz(3, 0, 0)).toMatchObject({ x: 3, y: 0, z: 0 });
      expect(xyz(4, 0, 0)).toMatchObject({ x: 4, y: 0, z: 0 });
      expect(xyz(5, 0, 0)).toMatchObject({ x: 5, y: 0, z: 0 });

      expect(xyz(0, 1, 0)).toMatchObject({ x: 0, y: 1, z: 0 });
      expect(xyz(1, 1, 0)).toMatchObject({ x: 1, y: 1, z: 0 });
      expect(xyz(2, 1, 0)).toMatchObject({ x: 2, y: 1, z: 0 });
      expect(xyz(3, 1, 0)).toMatchObject({ x: 3, y: 1, z: 0 });
      expect(xyz(4, 1, 0)).toMatchObject({ x: 4, y: 1, z: 0 });
      expect(xyz(5, 1, 0)).toMatchObject({ x: 5, y: 1, z: 0 });

      expect(xyz(0, 2, 0)).toMatchObject({ x: 0, y: 2, z: 0 });
      expect(xyz(1, 2, 0)).toMatchObject({ x: 1, y: 2, z: 0 });
      expect(xyz(2, 2, 0)).toMatchObject({ x: 2, y: 2, z: 0 });
      expect(xyz(3, 2, 0)).toMatchObject({ x: 3, y: 2, z: 0 });
      expect(xyz(4, 2, 0)).toMatchObject({ x: 4, y: 2, z: 0 });
      expect(xyz(5, 2, 0)).toMatchObject({ x: 5, y: 2, z: 0 });

      expect(xyz(0, 3, 0)).toMatchObject({ x: 0, y: 3, z: 0 });
      expect(xyz(1, 3, 0)).toMatchObject({ x: 1, y: 3, z: 0 });
      expect(xyz(2, 3, 0)).toMatchObject({ x: 2, y: 3, z: 0 });
      expect(xyz(3, 3, 0)).toMatchObject({ x: 3, y: 3, z: 0 });
      expect(xyz(4, 3, 0)).toMatchObject({ x: 4, y: 3, z: 0 });
      expect(xyz(5, 3, 0)).toMatchObject({ x: 5, y: 3, z: 0 });

      expect(xyz(0, 4, 0)).toMatchObject({ x: 0, y: 4, z: 0 });
      expect(xyz(1, 4, 0)).toMatchObject({ x: 1, y: 4, z: 0 });
      expect(xyz(2, 4, 0)).toMatchObject({ x: 2, y: 4, z: 0 });
      expect(xyz(3, 4, 0)).toMatchObject({ x: 3, y: 4, z: 0 });
      expect(xyz(4, 4, 0)).toMatchObject({ x: 4, y: 4, z: 0 });
      expect(xyz(5, 4, 0)).toMatchObject({ x: 5, y: 4, z: 0 });

      expect(xyz(0, 5, 0)).toMatchObject({ x: 0, y: 5, z: 0 });
      expect(xyz(1, 5, 0)).toMatchObject({ x: 1, y: 5, z: 0 });
      expect(xyz(2, 5, 0)).toMatchObject({ x: 2, y: 5, z: 0 });
      expect(xyz(3, 5, 0)).toMatchObject({ x: 3, y: 5, z: 0 });
      expect(xyz(4, 5, 0)).toMatchObject({ x: 4, y: 5, z: 0 });
      expect(xyz(5, 5, 0)).toMatchObject({ x: 5, y: 5, z: 0 });

      expect(xyz(0, 0, 1)).toMatchObject({ x: 0, y: 0, z: 1 });
      expect(xyz(1, 0, 1)).toMatchObject({ x: 1, y: 0, z: 1 });
      expect(xyz(2, 0, 1)).toMatchObject({ x: 2, y: 0, z: 1 });
      expect(xyz(3, 0, 1)).toMatchObject({ x: 3, y: 0, z: 1 });
      expect(xyz(4, 0, 1)).toMatchObject({ x: 4, y: 0, z: 1 });
      expect(xyz(5, 0, 1)).toMatchObject({ x: 5, y: 0, z: 1 });

      expect(xyz(0, 1, 1)).toMatchObject({ x: 0, y: 1, z: 1 });
      expect(xyz(1, 1, 1)).toMatchObject({ x: 1, y: 1, z: 1 });
      expect(xyz(2, 1, 1)).toMatchObject({ x: 2, y: 1, z: 1 });
      expect(xyz(3, 1, 1)).toMatchObject({ x: 3, y: 1, z: 1 });
      expect(xyz(4, 1, 1)).toMatchObject({ x: 4, y: 1, z: 1 });
      expect(xyz(5, 1, 1)).toMatchObject({ x: 5, y: 1, z: 1 });

      expect(xyz(0, 2, 1)).toMatchObject({ x: 0, y: 2, z: 1 });
      expect(xyz(1, 2, 1)).toMatchObject({ x: 1, y: 2, z: 1 });
      expect(xyz(2, 2, 1)).toMatchObject({ x: 2, y: 2, z: 1 });
      expect(xyz(3, 2, 1)).toMatchObject({ x: 3, y: 2, z: 1 });
      expect(xyz(4, 2, 1)).toMatchObject({ x: 4, y: 2, z: 1 });
      expect(xyz(5, 2, 1)).toMatchObject({ x: 5, y: 2, z: 1 });

      expect(xyz(0, 3, 1)).toMatchObject({ x: 0, y: 3, z: 1 });
      expect(xyz(1, 3, 1)).toMatchObject({ x: 1, y: 3, z: 1 });
      expect(xyz(2, 3, 1)).toMatchObject({ x: 2, y: 3, z: 1 });
      expect(xyz(3, 3, 1)).toMatchObject({ x: 3, y: 3, z: 1 });
      expect(xyz(4, 3, 1)).toMatchObject({ x: 4, y: 3, z: 1 });
      expect(xyz(5, 3, 1)).toMatchObject({ x: 5, y: 3, z: 1 });

      expect(xyz(0, 4, 1)).toMatchObject({ x: 0, y: 4, z: 1 });
      expect(xyz(1, 4, 1)).toMatchObject({ x: 1, y: 4, z: 1 });
      expect(xyz(2, 4, 1)).toMatchObject({ x: 2, y: 4, z: 1 });
      expect(xyz(3, 4, 1)).toMatchObject({ x: 3, y: 4, z: 1 });
      expect(xyz(4, 4, 1)).toMatchObject({ x: 4, y: 4, z: 1 });
      expect(xyz(5, 4, 1)).toMatchObject({ x: 5, y: 4, z: 1 });

      expect(xyz(0, 5, 1)).toMatchObject({ x: 0, y: 5, z: 1 });
      expect(xyz(1, 5, 1)).toMatchObject({ x: 1, y: 5, z: 1 });
      expect(xyz(2, 5, 1)).toMatchObject({ x: 2, y: 5, z: 1 });
      expect(xyz(3, 5, 1)).toMatchObject({ x: 3, y: 5, z: 1 });
      expect(xyz(4, 5, 1)).toMatchObject({ x: 4, y: 5, z: 1 });
      expect(xyz(5, 5, 1)).toMatchObject({ x: 5, y: 5, z: 1 });
    });
  });
});