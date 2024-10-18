import { readFileSync } from "node:fs";

import { seasonalities, initSync } from '../pkg';

import { describe, expect, it } from 'vitest';

initSync({ module: readFileSync('node_modules/@bsull/augurs/augurs_bg.wasm') });

describe('seasons', () => {
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

  it('works with number arrays', () => {
    expect(seasonalities(y)).toEqual(new Uint32Array([4]));
  });

  it('works with number arrays', () => {
    expect(seasonalities(new Float64Array(y))).toEqual(new Uint32Array([4]));
  });
});
