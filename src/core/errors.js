/** Typed errors with a stable code + user-facing action — UI never string-matches messages. */

export const CODES = Object.freeze({
  // capability
  E_INSECURE_CONTEXT: 'E_INSECURE_CONTEXT',
  E_NO_WEBGPU: 'E_NO_WEBGPU',
  E_NO_ADAPTER: 'E_NO_ADAPTER',
  E_NO_DEVICE: 'E_NO_DEVICE',
  E_NO_WEBCODECS: 'E_NO_WEBCODECS',
  E_NO_WASM: 'E_NO_WASM',
  E_PROBE_TIMEOUT: 'E_PROBE_TIMEOUT',
  // runtime / models
  E_ORT_LOAD_FAILED: 'E_ORT_LOAD_FAILED',
  E_ORT_SESSION_FAILED: 'E_ORT_SESSION_FAILED',
  E_TOKENIZER_FAILED: 'E_TOKENIZER_FAILED',
  // download / cache
  E_HTTP_STATUS: 'E_HTTP_STATUS',
  E_NETWORK_FAILED: 'E_NETWORK_FAILED',
  E_SIZE_MISMATCH: 'E_SIZE_MISMATCH',
  E_STORAGE_UNAVAILABLE: 'E_STORAGE_UNAVAILABLE',
  // pipeline
  E_CANCELLED: 'E_CANCELLED',
  E_ENCODE_FAILED: 'E_ENCODE_FAILED',
  E_INTERNAL: 'E_INTERNAL',
});

export const SEVERITY = Object.freeze({
  FATAL: 'fatal', // nothing can run
  DEGRADE: 'degrade', // runs, but slower / lower quality
  INFO: 'info',
});

export class SMError extends Error {
  constructor(code, message, { action = null, cause = null, detail = null } = {}) {
    super(message);
    this.name = 'SMError';
    this.code = code;
    this.action = action;
    this.detail = detail;
    if (cause != null) this.cause = cause;
  }
}

/** Wrap an unknown thrown value as an SMError, preserving a real SMError as-is. */
export function asSMError(err, code, message, opts = {}) {
  if (err instanceof SMError) return err;
  return new SMError(code, message, { ...opts, cause: err });
}

/** Throw a cancellation error if the signal is aborted. */
export function throwIfAborted(signal) {
  if (signal?.aborted) {
    throw new SMError(CODES.E_CANCELLED, 'Generation was cancelled.', { action: null });
  }
}

export function isCancel(err) {
  return err instanceof SMError && err.code === CODES.E_CANCELLED;
}
