import { webcrypto } from 'node:crypto'
import { readFileSync } from "node:fs";

import { PowerTransform, initSync } from '@bsull/augurs/transforms';

import { describe, expect } from 'vitest';

// Required for Rust's `rand::thread_rng` to support NodeJS modules.
// See https://docs.rs/getrandom#nodejs-es-module-support.
// @ts-ignore
globalThis.crypto = webcrypto

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

  describe('power transform', () => {
    const pt = new PowerTransform({ data: y });
    const transformed = pt.transform(y);
    expect(transformed).toBeInstanceOf(Float64Array);
    expect(transformed).toHaveLength(y.length);
    const inverse = pt.inverseTransform(transformed);
    expect(inverse).toBeInstanceOf(Float64Array);
    expect(inverse).toHaveLength(y.length);
    expect(new Array(inverse)).toEqual(y);
  })
})
