import {getImportMap} from "./getImportMap.js";
class WorkerWithImportMapViaBedfordsShim extends Worker {
  /**
   * @param {string | URL} scriptURL - The URL.
   * @param {WorkerOptions & {importMap?: 'inherit', debug?: boolean}} [options] - The options.
   */
  constructor(scriptURL, options = {}) {
    if (!options.importMap || options.importMap === 'inherit') {
      const shimURL = new URL('./WorkerWithImportMapViaBedfordsShim.worker.js', import.meta.url) + "";
      super(shimURL);
      const importMap = getImportMap();
      const baseURL = document.baseURI.split('/').slice(0, -1).join('/');
      if (options.debug) {
        console.log("WorkerWithImportMapViaBedfordsShim debug information", {importMap, shimURL, baseURL, options});
      }
      // Prevent error: URL object could not be cloned.
      scriptURL += '';
      this.postMessage({type: 'init-worker-with-import-map', importMap, scriptURL, baseURL, options});
    } else {
      super(scriptURL, options);
    }
  }
}
export {WorkerWithImportMapViaBedfordsShim};
