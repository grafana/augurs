import { readFileSync } from "node:fs";

import { Pipeline, initSync } from '@bsull/augurs/transforms';

import { describe, expect, it } from 'vitest';

initSync({ module: readFileSync('node_modules/@bsull/augurs/transforms_bg.wasm') });

describe('transforms', () => {
  const y = [
    0.1, 0.3, 0.8, 0.5,
    0.1, 0.31, 0.79, 0.48,
    0.09, 0.29, 0.81, 0.49,
    0.11, 0.28, 0.78, 0.53,
    0.1, 0.3, 0.8, 0.5,
    0.1, 0.31, 0.79, 0.48,
    0.09, 0.29, 0.81, 0.49,
    0.11, 0.28, 0.78, 0.53,
  ];

  expect.extend({
    toAllBeCloseTo: (received, expected) => {
      if (received.length !== expected.length) {
        return {
          message: () => `expected array lengths to match (got ${received.length}, wanted ${expected.length})`,
          pass: false,
        };
      }
      for (let index = 0; index < received.length; index++) {
        const got = received[index];
        const exp = expected[index];
        if (Math.abs(got - exp) > 0.1) {
          return {
            message: () => `got (${got}) not close to expected (${exp}) at index ${index}`,
            pass: false,
          }
        }
      }
      return { message: () => '', pass: true };
    }
  });


  describe('pipeline', () => {
    it('works with arrays', () => {
      const pt = new Pipeline(["standardScaler", "yeoJohnson"]);
      const transformed = pt.fitTransform(y);
      expect(transformed).toBeInstanceOf(Float64Array);
      expect(transformed).toHaveLength(y.length);
      console.log(transformed);
      const inverse = pt.inverseTransform(transformed);
      expect(inverse).toBeInstanceOf(Float64Array);
      expect(inverse).toHaveLength(y.length);
      //@ts-ignore
      expect(Array.from(inverse)).toAllBeCloseTo(y);
    });

    it('handles empty pipeline', () => {
      const pt = new Pipeline([]);
      expect(() => pt.fitTransform(y)).not.toThrow();
    });

    it('handles invalid transforms', () => {
      // @ts-ignore
      expect(() => new Pipeline(["invalidTransform"])).toThrow();
    });
  });
})
