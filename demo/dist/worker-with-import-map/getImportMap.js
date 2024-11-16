/**
 * @returns {object | undefined}
 */
function getImportMap() {
  /** @type {HTMLScriptElement|null} */
  const e = document.querySelector('script[type="importmap"]');
  if (!e?.textContent) {
    return;
  }
  return JSON.parse(e.textContent);
}
export {getImportMap};
