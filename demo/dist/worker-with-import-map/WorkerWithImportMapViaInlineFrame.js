import {EventHandler} from './EventHandler.js';
import {getImportMap} from './getImportMap.js';
window.workersReady = {};
class WorkerWithImportMapViaInlineFrame extends EventHandler {
  debug = false;
  iframe = document.createElement('iframe');
  callbackId = `cb${Math.floor(Math.random()*1000000000)}`;
  terminateId = `tm${Math.floor(Math.random()*1000000000)}`;
  /**
   * @param {URL | string} script - The worker URL.
   * @param {object} [options] - The options.
   * @param {object|'inherit'} [options.importMap] - The import map or simply `inherit`.
   * @returns 
   */
  constructor(script, options = {}) {
    super();
    const {iframe, callbackId, terminateId} = this;
    if (options.importMap === 'inherit') {
      options.importMap = getImportMap();
    }
    if (!options.importMap) {
      return new window.Worker(script, options);
    }
    window.workersReady[terminateId] = function(window) {
      iframe.remove();
    };
    this.ready = new Promise((resolve, reject) => {
      window.workersReady[callbackId] = function(window) {
        resolve();
      };
    });
    const html = `
<html>
  <head>
      <script type="importmap">${JSON.stringify(options.importMap)}</script>
  </head>
  <body onload="parent.workersReady.${callbackId}(this.window)">
    <script>
      ${EventHandler};
      class Self extends EventHandler {
        postMessage(e) {
          parent.postMessage(e);
        }
      };
      const self = new Self();
      window.self = self;
      window.onmessage = (e) => {
        self.dispatchEvent(e);
      };
    </script>
    <script type="module" src="${script}"></script>
  </body>
</html>`;
    if (!this.debug) {
      iframe.style.display = 'none';
    }
    document.body.appendChild(iframe);
    iframe.contentWindow.document.open();
    iframe.contentWindow.document.write(html);
    iframe.contentWindow.document.close();
    window.onmessage = (e) => {
      this.dispatchEvent(e);
    };
  }
  postMessage(data) {
    this.iframe.contentWindow.postMessage(data, '*');
  }
  terminate() {
    window.workersReady[this.terminateId]();
  }
}
export {WorkerWithImportMapViaInlineFrame};
