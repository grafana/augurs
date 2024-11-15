import {isBrowser, isJsDom} from 'browser-or-node';
import * as mod from 'module';
let internalRequire = null;
const ensureRequire = () => (!internalRequire) && (
  internalRequire = mod.createRequire(import.meta.url)
);
let ThisWorker = null;
if (isBrowser || isJsDom) {
  ThisWorker = (await import('./WorkerFrame.js')).WorkerFrame;
} else {
  ensureRequire();
  const NodeWorker = internalRequire('web-worker');
  function Worker(incomingPath, options = {}) {
    const filePath = incomingPath[0] === '.'
      ? new URL(incomingPath, options.root || import.meta.url).pathname
      : incomingPath;
    return new NodeWorker(filePath, options);
  };
  ThisWorker = Worker;
}
export const Worker = ThisWorker;
