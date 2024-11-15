export namespace AugursProphetWasmstanOptimizer {
  /**
   * Optimize the initial parameters given the data, returning the
   * optimal values under maximum likelihood estimation.
   */
  export function optimize(init: Inits, data: DataJson, opts: OptimizeOpts): OptimizeOutput;
}
import type { Inits } from './augurs-prophet-wasmstan-types.js';
export { Inits };
import type { DataJson } from './augurs-prophet-wasmstan-types.js';
export { DataJson };
import type { OptimizeOpts } from './augurs-prophet-wasmstan-types.js';
export { OptimizeOpts };
import type { OptimizeOutput } from './augurs-prophet-wasmstan-types.js';
export { OptimizeOutput };
