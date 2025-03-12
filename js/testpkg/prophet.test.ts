import { readFileSync } from "node:fs";
import {
  Prophet,
  ProphetHoliday,
  ProphetHolidayOccurrence,
  ProphetRegressor,
  ProphetSeasonality,
  ProphetSeasonalityOption,
  initSync
} from '@bsull/augurs/prophet';
import { optimizer } from '@bsull/augurs-prophet-wasmstan';
import { describe, expect, it } from 'vitest';

initSync({ module: readFileSync('node_modules/@bsull/augurs/prophet_bg.wasm') });

const START_TIMESTAMP = 1704067200;
const NINE_DAYS = 9 * 86400;
const TOTAL_LENGTH = 24;
const ds = Array.from({ length: TOTAL_LENGTH }, (_, i) => START_TIMESTAMP + (i * NINE_DAYS));
const y = Array.from({ length: TOTAL_LENGTH }, (_, i) => i + 1); // [1, 2, ..., 24]
const cap = Array(y.length).fill(24.0);
const floor = Array(y.length).fill(0.0);

describe('Prophet', () => {
  it('can be instantiated', () => {
    const prophet = new Prophet({ optimizer });
    expect(prophet).toBeInstanceOf(Prophet);
  });

  it('can be fit with arrays', () => {
    const prophet = new Prophet({ optimizer });
    prophet.fit({ ds, y });
    const preds = prophet.predict();
    expect(preds.yhat.point).toHaveLength(y.length);
  });

  it('can be fit with typed arrays', () => {
    const prophet = new Prophet({ optimizer });
    prophet.fit({ ds: new BigInt64Array(ds.map(BigInt)), y: new Float64Array(y) });
    const preds = prophet.predict();
    expect(preds.yhat.point).toHaveLength(y.length);
  });

  it('accepts cap/floor', () => {
    const prophet = new Prophet({ optimizer });
    prophet.fit({ ds, y, cap, floor });
    const preds = prophet.predict();
    expect(preds.yhat.point).toHaveLength(y.length);
  });

  it('returns regular arrays', () => {
    const prophet = new Prophet({ optimizer });
    prophet.fit({ ds, y });
    const preds = prophet.predict();
    expect(preds.yhat.point).toHaveLength(y.length);
    expect(preds.yhat.point).toBeInstanceOf(Array);
  });

  describe('holidays', () => {
    it('can be set', () => {
      const occurrences: ProphetHolidayOccurrence[] = [
        { start: new Date('2024-12-25').getTime() / 1000, end: new Date('2024-12-26').getTime() / 1000 },
      ];
      const holidays: Map<string, ProphetHoliday> = new Map([
        ["Christmas", { occurrences }],
      ]);
      new Prophet({ optimizer, holidays });
    });
  });

  describe('seasonality', () => {
    it('can be set', () => {
      const dailySeasonality: ProphetSeasonalityOption = { type: "manual", enabled: false };
      const prophet = new Prophet({ optimizer, dailySeasonality });
      const seasonality: ProphetSeasonality = { period: 30.5, fourierOrder: 5 };
      prophet.addSeasonality('daily', seasonality);
    });
  });

  describe('regressors', () => {
    it('can be set', () => {
      const reg: ProphetRegressor = { mode: "additive", priorScale: 0.5, standardize: "auto" };
      const prophet = new Prophet({ optimizer });
      prophet.addRegressor('feature1', reg);
    });

    it('can be set with just name', () => {
      const prophet = new Prophet({ optimizer });
      prophet.addRegressor('feature1');
    });

    it('can be set with name using mode', () => {
      const reg: ProphetRegressor = { mode: "multiplicative" };
      const prophet = new Prophet({ optimizer });
      prophet.addRegressor('feature1', reg);
    });
  });

  describe('makeFutureDataframe', () => {
    it('can be called with horizon without history', () => {
      const prophet = new Prophet({ optimizer });
      prophet.fit({ ds, y });

      const future = prophet.makeFutureDataframe(10, {includeHistory: false});
      expect(future.ds).toHaveLength(10);

      const fullSeries = [...ds, ...future.ds];
      for (let i = 1; i < fullSeries.length; i++) {
        expect(fullSeries[i] - fullSeries[i - 1]).toBe(NINE_DAYS);
      }
    });

    it('can be called with only horizon and no options argument', () => {
      const prophet = new Prophet({ optimizer });
      prophet.fit({ ds, y });

      const future = prophet.makeFutureDataframe(10);
      expect(future.ds).toHaveLength(10 + TOTAL_LENGTH);

      for (let i = 1; i < future.ds.length; i++) {
        expect(future.ds[i] - future.ds[i - 1]).toBe(NINE_DAYS);
      }
    });

    it('can be called with horizon and empty options argument', () => {
      const prophet = new Prophet({ optimizer });
      prophet.fit({ ds, y });

      const future = prophet.makeFutureDataframe(10, {});
      expect(future.ds).toHaveLength(10 + TOTAL_LENGTH);
    });
  });
});