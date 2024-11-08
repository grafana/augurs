import { readFileSync } from "node:fs";

import { AutoETS, initSync } from '@bsull/augurs/ets';

import { describe, expect, it } from 'vitest';

initSync({ module: readFileSync('node_modules/@bsull/augurs/ets_bg.wasm') });

const y = Array.from({ length: 20 }, (_, i) => i % 5);

describe('ets', () => {
  it('can be instantiated with a regular array', () => {
    const model = new AutoETS(10, 'ZZZ');
    expect(model).toBeInstanceOf(AutoETS);
  });

  it('can be fit', () => {
    const model = new AutoETS(5, 'ZZZ');
    model.fit(y);
  });

  it('can be fit with a typed array', () => {
    const model = new AutoETS(5, 'ZZZ');
    model.fit(new Float64Array(y));
  })

  it('returns regular arrays', () => {
    const model = new AutoETS(5, 'ZZZ');
    model.fit(new Float64Array(y));
    expect(model.predict(10).point).toBeInstanceOf(Array);
  });

  it('returns regular arrays for intervals', () => {
    const model = new AutoETS(5, 'ZZZ');
    model.fit(new Float64Array(y));
    const level = 0.95;
    const prediction = model.predict(10, level);
    expect(prediction.point).toBeInstanceOf(Array);
    expect(prediction.point).toHaveLength(10);
    expect(prediction.intervals?.level).toEqual(level);
    expect(prediction.intervals?.lower).toBeInstanceOf(Array);
    expect(prediction.intervals?.upper).toBeInstanceOf(Array);
    expect(prediction.intervals?.lower).toHaveLength(10);
    expect(prediction.intervals?.upper).toHaveLength(10);
  })
})

