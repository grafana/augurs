import { readFileSync } from "node:fs";

import { Dtw, initSync } from '@bsull/augurs/dtw';

import { describe, expect, it } from 'vitest';

initSync({ module: readFileSync('node_modules/@bsull/augurs/dtw_bg.wasm') });

describe('dtw', () => {
  describe('distance', () => {
    describe('euclidean', () => {
      it('can be instantiated with a constructor', () => {
        const dtw = new Dtw('euclidean');
        expect(dtw).toBeInstanceOf(Dtw);
      });

      it('can be instantiated with a static method', () => {
        const dtw = Dtw.euclidean();
        expect(dtw).toBeInstanceOf(Dtw);
      });

      it('can be instantiated with a custom window', () => {
        const dtw = Dtw.euclidean({ window: 10 });
        expect(dtw).toBeInstanceOf(Dtw);
      });

      it('can be instantiated with a custom max distance', () => {
        const dtw = Dtw.euclidean({ maxDistance: 10.0 });
        expect(dtw).toBeInstanceOf(Dtw);
      });

      it('can be instantiated with a custom lower bound', () => {
        const dtw = Dtw.euclidean({ lowerBound: 10.0 });
        expect(dtw).toBeInstanceOf(Dtw);
      });

      it('can be instantiated with a custom upper bound', () => {
        const dtw = Dtw.euclidean({ upperBound: 10.0 });
        expect(dtw).toBeInstanceOf(Dtw);
      });

      // Commented out because we compile without parallelism for now.
      // it('can be instantiated with a custom parallelize', () => {
      //   const dtw = Dtw.euclidean({ parallelize: true });
      //   expect(dtw).toBeInstanceOf(Dtw);
      // });

      it('can be fit with number arrays', () => {
        const dtw = Dtw.euclidean();
        expect(dtw.distance([0.0, 1.0, 2.0], [3.0, 4.0, 5.0])).toBeCloseTo(5.0990195135927845);
      });

      it('can be fit with typed arrays', () => {
        const dtw = Dtw.euclidean();
        expect(dtw.distance(new Float64Array([0.0, 1.0, 2.0]), new Float64Array([3.0, 4.0, 5.0]))).toBeCloseTo(5.0990195135927845);
      });

      it('can be fit with different length arrays', () => {
        const dtw = Dtw.euclidean();
        expect(dtw.distance([0.0, 1.0, 2.0], [3.0, 4.0, 5.0, 6.0])).toBeCloseTo(6.48074069840786);
      });

      it('can be fit with empty arrays', () => {
        const dtw = Dtw.euclidean();
        expect(dtw.distance([], [3.0, 4.0, 5.0])).toBe(Infinity);
        expect(dtw.distance([3.0, 4.0, 5.0], [])).toBe(Infinity);
      });

      it('gives a useful error when passed the wrong kind of data', () => {
        const dtw = Dtw.euclidean();
        // @ts-expect-error
        expect(() => dtw.distance(['hi', 2, 3], [4, 5, 6])).toThrowError('TypeError: expected array of numbers or Float64Array');
      })
    });

    describe('manhattan', () => {
      it('can be instantiated with a constructor', () => {
        const dtw = new Dtw('manhattan');
        expect(dtw).toBeInstanceOf(Dtw);
      });

      it('can be instantiated with a static method', () => {
        const dtw = Dtw.manhattan();
        expect(dtw).toBeInstanceOf(Dtw);
      });

      it('can be fit with number arrays', () => {
        const dtw = Dtw.manhattan();
        expect(dtw.distance(
          [0., 0., 1., 2., 1., 0., 1., 0., 0.],
          [0., 1., 2., 0., 0., 0., 0., 0., 0.],
        )).toBeCloseTo(2.0);
      });

      it('can be fit with typed arrays', () => {
        const dtw = Dtw.manhattan();
        expect(dtw.distance(
          new Float64Array([0., 0., 1., 2., 1., 0., 1., 0., 0.]),
          new Float64Array([0., 1., 2., 0., 0., 0., 0., 0., 0.]),
        )).toBeCloseTo(2.0);
      });

      it('gives a useful error when passed the wrong kind of data', () => {
        const dtw = Dtw.manhattan();
        // @ts-expect-error
        expect(() => dtw.distance(['hi', 2, 3], [4, 5, 6])).toThrowError('TypeError: expected array of numbers or Float64Array');
      });
    });
  });

  describe('distanceMatrix', () => {
    it('can be fit with number arrays', () => {
      const dtw = Dtw.euclidean();
      const series = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]];
      const dists = dtw.distanceMatrix(series);
      expect(dists).toHaveLength(3);
      expect(dists[0]).toHaveLength(3);
      expect(dists[0][0]).toBeCloseTo(0.0);
      expect(dists[0][1]).toBeCloseTo(5.0990195135927845);
      expect(dists[0][2]).toBeCloseTo(10.392304845413264);
    });

    it('can be fit with typed arrays', () => {
      const dtw = Dtw.euclidean();
      const series = [new Float64Array([0.0, 1.0, 2.0]), new Float64Array([3.0, 4.0, 5.0]), new Float64Array([6.0, 7.0, 8.0])];
      const dists = dtw.distanceMatrix(series);
      expect(dists).toBeInstanceOf(Array);
      expect(dists).toHaveLength(3);
      expect(dists[0]).toBeInstanceOf(Float64Array);
      expect(dists[0]).toHaveLength(3);
      expect(dists[0][0]).toBeCloseTo(0.0);
      expect(dists[0][1]).toBeCloseTo(5.0990195135927845);
      expect(dists[0][2]).toBeCloseTo(10.392304845413264);
    });

    it('gives a useful error when passed the wrong kind of data', () => {
      const dtw = Dtw.euclidean();
      // @ts-expect-error
      expect(() => dtw.distanceMatrix([1, 2, 3])).toThrowError('TypeError: expected array of number arrays or array of Float64Array');
    });
  });
});
