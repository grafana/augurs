import { readFileSync } from "node:fs";

import { ChangepointDetector, initSync } from '@bsull/augurs/changepoint';

import { describe, expect, it } from 'vitest';

initSync({ module: readFileSync('node_modules/@bsull/augurs/changepoint_bg.wasm') });

describe('changepoints', () => {
  const y = [0.5, 1.0, 0.4, 0.8, 1.5, 0.9, 0.6, 25.3, 20.4, 27.3, 30.0];

  describe('normalGamma', () => {
    it('can be used with a constructor', () => {
      const detector = new ChangepointDetector('normal-gamma');
      expect(detector).toBeInstanceOf(ChangepointDetector);
      const cps = detector.detectChangepoints(y);
      expect(cps.indices).toEqual([0, 6]);
    });

    it('can be used with a static method', () => {
      const detector = ChangepointDetector.normalGamma();
      expect(detector).toBeInstanceOf(ChangepointDetector);
      const cps = detector.detectChangepoints(y);
      expect(cps.indices).toEqual([0, 6]);
    });

    it('can be used with typed arrays', () => {
      const detector = ChangepointDetector.normalGamma();
      expect(detector).toBeInstanceOf(ChangepointDetector);
      const cps = detector.detectChangepoints(new Float64Array(y));
      expect(cps.indices).toEqual([0, 6]);
    });
  });

  describe('defaultArgpcp', () => {
    it('can be used with a constructor', () => {
      const detector = new ChangepointDetector('default-argpcp');
      expect(detector).toBeInstanceOf(ChangepointDetector);
      const cps = detector.detectChangepoints(y);
      expect(cps.indices).toEqual([0, 6]);
    });

    it('can be used with a static method', () => {
      const detector = ChangepointDetector.defaultArgpcp();
      expect(detector).toBeInstanceOf(ChangepointDetector);
      const cps = detector.detectChangepoints(y);
      expect(cps.indices).toEqual([0, 6]);
    });

    it('can be used with typed arrays', () => {
      const detector = ChangepointDetector.defaultArgpcp();
      expect(detector).toBeInstanceOf(ChangepointDetector);
      const cps = detector.detectChangepoints(new Float64Array(y));
      expect(cps.indices).toEqual([0, 6]);
    });
  });
});

