export namespace WasiFilesystemPreopens {
  export function getDirectories(): Array<[Descriptor, string]>;
}
import type { Descriptor } from './wasi-filesystem-types.js';
export { Descriptor };
