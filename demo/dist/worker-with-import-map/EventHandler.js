class EventHandler {
  /** @type {Function|null} */
  onclick = null;
  /** @type {Function|null} */
  onmessage = null;
  /** @type {Record<string, Set<Function>>} */
  events = {};
  /**
   * @param {string} type - The event type.
   * @param {Function} cb - The callback.
   */
  addEventListener(type, cb) {
    this.events[type] ??= new Set();
    this.events[type].add(cb);
  }
  /**
   * @param {string} type - The event type.
   * @param {Function} cb - The callback.
   */
  removeEventListener(type, cb) {
    this.events[type]?.delete(cb);
  }
  /**
   * @param {Event} event - The event.
   */
  dispatchEvent(event) {
    const {type} = event;
    this['on' + type]?.(event);
    this.events[type]?.forEach(listener => listener(event));
  }
}
export {EventHandler};
