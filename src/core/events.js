/** Tiny typed event emitter — pipeline reports progress to UI without either importing the other. */
export class Emitter {
  constructor() {
    this._map = new Map();
  }

  on(type, fn) {
    if (!this._map.has(type)) this._map.set(type, new Set());
    this._map.get(type).add(fn);
    return () => this.off(type, fn);
  }

  off(type, fn) {
    this._map.get(type)?.delete(fn);
  }

  emit(type, payload) {
    for (const fn of this._map.get(type) ?? []) {
      try {
        fn(payload);
      } catch (err) {
        console.error(`listener for "${type}" threw`, err);
      }
    }
  }
}
