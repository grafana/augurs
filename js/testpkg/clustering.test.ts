import { readFileSync } from "node:fs";

import { DbscanClusterer, initSync as initClusteringSync } from '@bsull/augurs/clustering';
import { Dtw, initSync as initDtwSync } from '@bsull/augurs/dtw';

import { describe, expect, it } from 'vitest';

initClusteringSync({ module: readFileSync('node_modules/@bsull/augurs/clustering_bg.wasm') });
initDtwSync({ module: readFileSync('node_modules/@bsull/augurs/dtw_bg.wasm') });

describe('clustering', () => {
  it('can be instantiated', () => {
    const clusterer = new DbscanClusterer({ epsilon: 0.5, minClusterSize: 2 });
    expect(clusterer).toBeInstanceOf(DbscanClusterer);
  });

  it('can be fit with a raw distance matrix of number arrays', () => {
    const clusterer = new DbscanClusterer({ epsilon: 1.0, minClusterSize: 2 });
    const labels = clusterer.fit([
      [0, 1, 2, 3],
      [1, 0, 3, 3],
      [2, 3, 0, 4],
      [3, 3, 4, 0],
    ]);
    expect(labels).toEqual(new Int32Array([1, 1, -1, -1]));
  });

  it('can be fit with a raw distance matrix of typed arrays', () => {
    const clusterer = new DbscanClusterer({ epsilon: 1.0, minClusterSize: 2 });
    const labels = clusterer.fit([
      new Float64Array([0, 1, 2, 3]),
      new Float64Array([1, 0, 3, 3]),
      new Float64Array([2, 3, 0, 4]),
      new Float64Array([3, 3, 4, 0]),
    ]);
    expect(labels).toEqual(new Int32Array([1, 1, -1, -1]));
  });

  it('can be fit with a distance matrix from augurs', () => {
    const dtw = Dtw.euclidean();
    const distanceMatrix = dtw.distanceMatrix([
      [1, 3, 4],
      [1, 3, 3.9],
      [1.1, 2.9, 4.1],
      [5, 6.2, 10],
    ]);
    const clusterer = new DbscanClusterer({ epsilon: 0.5, minClusterSize: 2 });
    const labels = clusterer.fit(distanceMatrix);
    expect(labels).toEqual(new Int32Array([1, 1, 1, -1]));
  })
});
