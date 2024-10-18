import { readFileSync } from "node:fs";

import { initLogging, initSync } from '../pkg';

import { describe, expect, it } from 'vitest';

initSync({ module: readFileSync('node_modules/@bsull/augurs/augurs_bg.wasm') });

describe('logging', () => {
  it('can be initialized with no config', () => {
    expect(() => initLogging()).not.toThrow();
  });

  it('can be initialized with a config', () => {
    // Note: this will throw because there's a global default logger which can't
    // be unset for performance reasons.
    // We mainly just want to make sure that the config is accepted.
    expect(() => initLogging({ maxLevel: 'info', target: 'console', color: true }))
      .toThrow("logging already initialized");
  });

  it('can be initialized multiple times without panicking', () => {
    // These will throw exceptions but at least they're not panics.
    expect(() => initLogging()).toThrow("logging already initialized");
    expect(() => initLogging()).toThrow("logging already initialized");
  });
})
