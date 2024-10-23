import { readFileSync } from "node:fs";

import { OutlierDetector, initSync } from '@bsull/augurs/outlier';

import { describe, expect, it } from 'vitest';

initSync({ module: readFileSync('node_modules/@bsull/augurs/outlier_bg.wasm') });

describe('dbscan', () => {

  it('can be instantiated with a constructor', () => {
    const detector = new OutlierDetector('dbscan', { sensitivity: 0.5 });
    expect(detector).toBeInstanceOf(OutlierDetector);
  });

  it('can be instantiated with a static method', () => {
    const detector = OutlierDetector.dbscan({ sensitivity: 0.5 });
    expect(detector).toBeInstanceOf(OutlierDetector);
  });

  it('gives a useful error when passed the wrong kind of data', () => {
    const detector = OutlierDetector.dbscan({ sensitivity: 0.5 });
    // @ts-expect-error
    expect(() => detector.detect([1, 3, 4])).toThrowError('TypeError: expected array of number arrays');
  })

  it('detects outliers using the concise API', () => {
    const detector = OutlierDetector.dbscan({ sensitivity: 0.5 });
    const outliers = detector.detect([
      [1, 3, 4],
      [1, 3, 3.9],
      [1.1, 2.9, 4.1],
      [1, 2.9, 10],
    ]);
    expect(outliers.outlyingSeries).toEqual([3]);
    // expect(outliers.clusterBand).toEqual({ min: [1, 3, 4.1], max: [1.1, 3, 4.1] });
    expect(outliers.seriesResults).toHaveLength(4);
    expect(outliers.seriesResults[0].isOutlier).toBe(false);
    expect(outliers.seriesResults[0].scores).toEqual([0.0, 0.0, 0.0]);
    expect(outliers.seriesResults[1].isOutlier).toBe(false);
    expect(outliers.seriesResults[1].scores).toEqual([0.0, 0.0, 0.0]);
    expect(outliers.seriesResults[2].isOutlier).toBe(false);
    expect(outliers.seriesResults[2].scores).toEqual([0.0, 0.0, 0.0]);
    expect(outliers.seriesResults[3].isOutlier).toBe(true);
    expect(outliers.seriesResults[3].scores).toEqual([0.0, 0.0, 1.0]);
    expect(outliers.seriesResults[3].outlierIntervals).toEqual([{ start: 2, end: undefined }]);
  });

  it('concise API works with typed arrays', () => {
    const detector = OutlierDetector.dbscan({ sensitivity: 0.5 });
    const outliers = detector.detect([
      new Float64Array([1, 3, 4]),
      new Float64Array([1, 3, 3.9]),
      new Float64Array([1.1, 2.9, 4.1]),
      new Float64Array([1, 2.9, 10]),
    ]);
    expect(outliers.outlyingSeries).toEqual([3]);
    // expect(outliers.clusterBand).toEqual({ min: [1, 3, 4.1], max: [1.1, 3, 4.1] });
    expect(outliers.seriesResults).toHaveLength(4);
    expect(outliers.seriesResults[0].isOutlier).toBe(false);
    expect(outliers.seriesResults[0].scores).toEqual([0.0, 0.0, 0.0]);
    expect(outliers.seriesResults[1].isOutlier).toBe(false);
    expect(outliers.seriesResults[1].scores).toEqual([0.0, 0.0, 0.0]);
    expect(outliers.seriesResults[2].isOutlier).toBe(false);
    expect(outliers.seriesResults[2].scores).toEqual([0.0, 0.0, 0.0]);
    expect(outliers.seriesResults[3].isOutlier).toBe(true);
    expect(outliers.seriesResults[3].scores).toEqual([0.0, 0.0, 1.0]);
    expect(outliers.seriesResults[3].outlierIntervals).toEqual([{ start: 2, end: undefined }]);
  });

  it('can be preloaded and run multiple times', () => {
    const detector = OutlierDetector.dbscan({ sensitivity: 0.5 });
    const loaded = detector.preprocess([
      [1, 3, 4],
      [1, 3, 3.9],
      [1.1, 2.9, 4.1],
      [1, 2.9, 10],
    ]);
    let outliers = loaded.detect();
    expect(outliers.outlyingSeries).toEqual([3]);
    // expect(outliers.clusterBand).toEqual({ min: [1, 3, 4.1], max: [1.1, 3, 4.1] });

    loaded.updateDetector({ sensitivity: 0.01 });
    outliers = loaded.detect();
    expect(outliers.outlyingSeries).toEqual([]);
    // expect(outliers.clusterBand).toEqual({ min: [1, 3, 4.1], max: [1.1, 3, 4.1] });
  });
});

describe('mad', () => {
  it('can be instantiated', () => {
    const detector = OutlierDetector.mad({ sensitivity: 0.5 });
    expect(detector).toBeInstanceOf(OutlierDetector);
  });

  it('gives a useful error when passed the wrong kind of data', () => {
    const detector = OutlierDetector.mad({ sensitivity: 0.5 });
    // @ts-expect-error
    expect(() => detector.detect([1, 3, 4])).toThrowError('TypeError: expected array of number arrays');
  })

  it('detects outliers using the concise API', () => {
    const detector = OutlierDetector.mad({ sensitivity: 0.5 });
    const outliers = detector.detect([
      [1, 3, 4],
      [1, 3, 3.9],
      [1.1, 2.9, 4.1],
      [1, 2.9, 10],
    ]);
    expect(outliers.outlyingSeries).toEqual([3]);
    // expect(outliers.clusterBand).toEqual({ min: [1, 3, 4.1], max: [1.1, 3, 4.1] });
    expect(outliers.seriesResults).toHaveLength(4);
    expect(outliers.seriesResults[0].isOutlier).toBe(false);
    expect(outliers.seriesResults[1].isOutlier).toBe(false);
    expect(outliers.seriesResults[2].isOutlier).toBe(false);
    expect(outliers.seriesResults[3].isOutlier).toBe(true);
    expect(outliers.seriesResults[3].outlierIntervals).toEqual([{ start: 2, end: undefined }]);
  });

  it('concise API works with typed arrays', () => {
    const detector = OutlierDetector.mad({ sensitivity: 0.5 });
    const outliers = detector.detect([
      new Float64Array([1, 3, 4]),
      new Float64Array([1, 3, 3.9]),
      new Float64Array([1.1, 2.9, 4.1]),
      new Float64Array([1, 2.9, 10]),
    ]);
    expect(outliers.outlyingSeries).toEqual([3]);
    // expect(outliers.clusterBand).toEqual({ min: [1, 3, 4.1], max: [1.1, 3, 4.1] });
    expect(outliers.seriesResults).toHaveLength(4);
    expect(outliers.seriesResults[0].isOutlier).toBe(false);
    expect(outliers.seriesResults[1].isOutlier).toBe(false);
    expect(outliers.seriesResults[2].isOutlier).toBe(false);
    expect(outliers.seriesResults[3].isOutlier).toBe(true);
    expect(outliers.seriesResults[3].outlierIntervals).toEqual([{ start: 2, end: undefined }]);
  });

  it('can be preloaded and run multiple times', () => {
    const detector = OutlierDetector.mad({ sensitivity: 0.5 });
    const loaded = detector.preprocess([
      [1, 3, 4],
      [1, 3, 3.9],
      [1.1, 2.9, 4.1],
      [1, 2.9, 10],
    ]);
    let outliers = loaded.detect();
    expect(outliers.outlyingSeries).toEqual([3]);
    // expect(outliers.clusterBand).toEqual({ min: [1, 3, 4.1], max: [1.1, 3, 4.1] });

    loaded.updateDetector({ sensitivity: 0.01 });
    outliers = loaded.detect();
    expect(outliers.outlyingSeries).toEqual([]);
    // expect(outliers.clusterBand).toEqual({ min: [1, 3, 4.1], max: [1.1, 3, 4.1] });
  });
});
