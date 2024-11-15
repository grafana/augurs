const shimCodeUrl = "https://ga.jspm.io/npm:es-module-shims@1.6.2/dist/es-module-shims.wasm.js";
/**
 * @param {string} baseURL - The base URL based on `document.baseURI`
 * @param {string} path - The path that may be relative.
 * @param {boolean} debug - Whether to print debug output.
 * @returns {string} - The new absolute path.
 */
function makeRelativePath(baseURL, path, debug) {
  if (path[0] !== '.') {
    return path;
  }
  const ret = baseURL + '/' + path;
  if (debug) {
    console.log(`makeRelativePath> Update path from ${path} to ${ret}`);
  }
  return ret;
}
self.addEventListener('message', function(e) {
  if (e.data.type && e.data.type === 'init-worker-with-import-map') {
    const {scriptURL, importMap, baseURL, options} = e.data;
    if (options.debug) {
      console.log('Got init-worker-with-import-map message');
    }
    /** @param {string} path - Relative path */
    const fixPath = (path) => makeRelativePath(baseURL, path, options.debug);
    if (importMap?.imports) {
      const {imports} = importMap;
      for (const key in imports) {
        imports[key] = fixPath(imports[key]);
      }
    }
    importScripts(shimCodeUrl);
    importShim.addImportMap(importMap);
    const absoluteScriptURL = fixPath(scriptURL);
    importShim(absoluteScriptURL)
      .then(() => {
        if (options.debug) {
          console.log(`${absoluteScriptURL} worker has been loaded`);
        }
      })
      .catch(e => {
        if (options.debug) {
          debugger;
        }
        setTimeout(() => {
          throw e;
        })
      });
  }
});
