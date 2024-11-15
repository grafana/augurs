export namespace WasiIoStreams {
  export { InputStream };
  export { OutputStream };
}
import type { Error } from './wasi-io-error.js';
export { Error };
export type StreamError = StreamErrorLastOperationFailed | StreamErrorClosed;
export interface StreamErrorLastOperationFailed {
  tag: 'last-operation-failed',
  val: Error,
}
export interface StreamErrorClosed {
  tag: 'closed',
}

export class InputStream {
  read(len: bigint): Uint8Array;
  blockingRead(len: bigint): Uint8Array;
}

export class OutputStream {
  checkWrite(): bigint;
  write(contents: Uint8Array): void;
  blockingWriteAndFlush(contents: Uint8Array): void;
  blockingFlush(): void;
}
