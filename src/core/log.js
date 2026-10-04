/** Minimal structured logger with child scopes + in-memory ring buffer (survives tab-close, console doesn't). */

const LEVELS = { debug: 10, info: 20, warn: 30, error: 40 };
const RING_MAX = 500;

class Logger {
  constructor(scope = 'app', shared = { ring: [], level: 20, listeners: new Set() }) {
    this.scope = scope;
    this._s = shared;
  }

  child(scope) {
    return new Logger(this.scope === 'app' ? scope : `${this.scope}.${scope}`, this._s);
  }

  setLevel(name) {
    if (name in LEVELS) this._s.level = LEVELS[name];
  }

  /** Subscribe to records as they arrive. Returns unsubscribe. */
  subscribe(fn) {
    this._s.listeners.add(fn);
    return () => this._s.listeners.delete(fn);
  }

  dump() {
    return this._s.ring.slice();
  }

  _emit(level, message, data) {
    if (LEVELS[level] < this._s.level) return;
    const rec = { t: Date.now(), level, scope: this.scope, message, data: data ?? null };
    this._s.ring.push(rec);
    if (this._s.ring.length > RING_MAX) this._s.ring.shift();
    for (const fn of this._s.listeners) {
      try {
        fn(rec);
      } catch {
        /* a broken listener must not break logging */
      }
    }
    const line = `[${this.scope}] ${message}`;
    if (level === 'error') console.error(line, data ?? '');
    else if (level === 'warn') console.warn(line, data ?? '');
    else if (level === 'debug') console.debug(line, data ?? '');
    else console.log(line, data ?? '');
  }

  debug(m, d) { this._emit('debug', m, d); }
  info(m, d) { this._emit('info', m, d); }
  warn(m, d) { this._emit('warn', m, d); }
  error(m, d) { this._emit('error', m, d); }
}

export const logger = new Logger();
