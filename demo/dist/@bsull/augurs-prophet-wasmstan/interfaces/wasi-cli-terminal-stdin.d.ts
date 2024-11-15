export namespace WasiCliTerminalStdin {
  export function getTerminalStdin(): TerminalInput | undefined;
}
import type { TerminalInput } from './wasi-cli-terminal-input.js';
export { TerminalInput };
