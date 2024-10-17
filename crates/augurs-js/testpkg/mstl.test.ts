import { readFileSync } from "node:fs";

import { ets, MSTL, initSync } from '../pkg';

import { describe, expect, it } from 'vitest';

initSync({ module: readFileSync('node_modules/@bsull/augurs/augurs_bg.wasm') });

const y = Array.from({ length: 20 }, (_, i) => i % 5);

describe('ets', () => {

  it('can be instantiated with a constructor', () => {
    const model = new MSTL('ets', [10]);
    expect(model).toBeInstanceOf(MSTL);
  });

  it('can be instantiated with a static method', () => {
    const model = MSTL.ets([10]);
    expect(model).toBeInstanceOf(MSTL);
  });

  it('can be instantiated with a standalone function', () => {
    // @ts-ignore
    const model = ets([10]);
    expect(model).toBeInstanceOf(MSTL);
  });

  it('can be instantiated with a regular array', () => {
    const model = MSTL.ets([10]);
    expect(model).toBeInstanceOf(MSTL);
  });

  it('can be instantiated with a typed array', () => {
    const model = MSTL.ets(new Uint32Array([7]));
    expect(model).toBeInstanceOf(MSTL);
  });

  it('can be fit', () => {
    const model = MSTL.ets([5]);
    model.fit(y);
  });

  it('can be fit with a typed array', () => {
    const model = MSTL.ets(new Uint32Array([5]));
    model.fit(new Float64Array(y));
  })

  it('returns regular arrays', () => {
    const model = MSTL.ets(new Uint32Array([5]));
    model.fit(new Float64Array(y));
    expect(model.predictInSample().point).toBeInstanceOf(Array);
  });

  it('returns regular arrays for intervals', () => {
    const model = MSTL.ets(new Uint32Array([5]));
    model.fit(new Float64Array(y));
    const level = 0.95;
    expect(model.predictInSample(level).point).toBeInstanceOf(Array);
    expect(model.predictInSample(level).intervals!.lower).toBeInstanceOf(Array);
    expect(model.predictInSample(level).intervals!.upper).toBeInstanceOf(Array);
    expect(model.predictInSample(level).intervals!.level).toEqual(level);
  })
})
