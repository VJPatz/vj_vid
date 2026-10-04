/**
 * Shared ONNX Runtime Web loader — one module owns `ort.env` config.
 * wasmPaths -> /ort/ (self-hosted, see copy-ort-assets.mjs) to stay offline.
 * numThreads=1 (no COOP/COEP on a static host). WebGPU preferred, WASM fallback —
 * also retried at run() time, since some kernels fail there even after a clean
 * session create.
 */

import { CODES, SMError, asSMError } from '../core/errors.js';
import { logger } from '../core/log.js';

const BASE_HREF =
  typeof document !== 'undefined' && document.baseURI
    ? document.baseURI
    : typeof self !== 'undefined' && self.location
      ? self.location.href
      : 'http://localhost/';
const ORT_WASM_PATH = new URL('./ort/', BASE_HREF).href;

let ortPromise = null;
let configured = false;

export function loadOrt() {
  if (!ortPromise) {
    ortPromise = (async () => {
      let ort;
      try {
        // Plain 'onnxruntime-web' doesn't register WebGPU in 1.17.1 — must
        // import this subpath, or WebGPU silently falls through to WASM.
        ort = await import('onnxruntime-web/webgpu');
      } catch (err) {
        throw asSMError(err, CODES.E_ORT_LOAD_FAILED, 'Could not load ONNX Runtime Web.', {
          action: 'Run `npm install`, then reload. If it persists: `rm -rf node_modules && npm install`.',
        });
      }
      if (!configured) {
        try {
          ort.env.wasm.wasmPaths = ORT_WASM_PATH;
          ort.env.wasm.numThreads = 1;
          ort.env.wasm.simd = true;
          ort.env.logLevel = 'error';
          configured = true;
        } catch (err) {
          logger.child('ort').warn('could not fully configure ORT env', { error: String(err?.message ?? err) });
        }
      }
      logger.child('ort').info('runtime ready', { wasmPaths: ORT_WASM_PATH });
      return ort;
    })();
  }
  return ortPromise;
}

export function webgpuAvailable() {
  return typeof navigator !== 'undefined' && Boolean(navigator.gpu);
}

async function createRawSession(ort, bytes, provider, freeDimensionOverrides) {
  return ort.InferenceSession.create(bytes, {
    executionProviders: [provider],
    graphOptimizationLevel: 'all',
    executionMode: 'sequential', // lower peak memory, matters more than latency here
    ...(freeDimensionOverrides ? { freeDimensionOverrides } : {}),
  });
}

/**
 * Create an InferenceSession, preferring WebGPU.
 * `freeDimensionOverrides` pins symbolic dims at creation (batch/height/width/seq_len)
 * — required for this model repo's exports to pick the right kernel variant.
 * `.run()` transparently falls back to WASM (and stays there) if a WebGPU kernel
 * throws at run time, not just at creation.
 */
export async function createSession({ bytes, label, preferWebgpu = true, freeDimensionOverrides = null }) {
  const ort = await loadOrt();
  const attempts = [];
  if (preferWebgpu && webgpuAvailable()) attempts.push('webgpu');
  attempts.push('wasm');

  let raw = null;
  let provider = null;
  let lastError = null;
  for (const p of attempts) {
    try {
      raw = await createRawSession(ort, bytes, p, freeDimensionOverrides);
      provider = p;
      logger.child('ort').info('session created', { label, provider });
      break;
    } catch (err) {
      lastError = err;
      logger.child('ort').warn('session creation failed', { label, provider: p, error: String(err?.message ?? err) });
    }
  }
  if (!raw) {
    throw new SMError(CODES.E_ORT_SESSION_FAILED, `Could not create an inference session for "${label}".`, {
      cause: lastError,
      action: 'The downloaded model file may be corrupt. Clear the model cache in Settings and try again.',
      detail: { label, attempted: attempts, error: String(lastError?.message ?? lastError) },
    });
  }

  const state = { session: raw, provider };
  const wrapper = {
    get inputNames() { return state.session.inputNames; },
    get outputNames() { return state.session.outputNames; },
    async run(feeds, opts) {
      try {
        return await state.session.run(feeds, opts);
      } catch (err) {
        if (state.provider !== 'webgpu') throw err; // already on wasm, nothing left to fall back to
        logger.child('ort').warn('session.run failed on webgpu, retrying on wasm', {
          label, error: String(err?.message ?? err),
        });
        const retry = await createRawSession(ort, bytes, 'wasm', freeDimensionOverrides);
        try { await state.session.release?.(); } catch { /* best effort */ }
        state.session = retry;
        state.provider = 'wasm';
        return await state.session.run(feeds, opts);
      }
    },
    async release() { return state.session.release?.(); },
  };
  return { session: wrapper, get provider() { return state.provider; } };
}

export async function makeIntTensor(values, dims, use64 = true) {
  const ort = await loadOrt();
  if (use64) return new ort.Tensor('int64', BigInt64Array.from(values, (v) => BigInt(v)), dims);
  return new ort.Tensor('int32', Int32Array.from(values, (v) => Number(v)), dims);
}

export async function makeFloatTensor(values, dims) {
  const ort = await loadOrt();
  const data = values instanceof Float32Array ? values : Float32Array.from(values);
  return new ort.Tensor('float32', data, dims);
}

export async function releaseSession(session, label = 'session') {
  if (!session) return;
  try {
    await session.release?.();
  } catch (err) {
    logger.child('ort').warn('session release failed', { label, error: String(err?.message ?? err) });
  }
}
