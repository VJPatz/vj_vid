/**
 * Runtime capability probe. Acquires the real thing then discards it —
 * feature-detect alone (`typeof navigator.gpu`) passes on machines where
 * requestDevice()/encoding still fails. Resolves, never rejects; finishes
 * within budgetMs; leaves nothing behind; every miss is a typed reason.
 */

import { CODES, SEVERITY } from '../core/errors.js';

export const TIER = Object.freeze({ FULL: 'full', STANDARD: 'standard', NONE: 'none' });

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const reason = (code, severity, message, action, detail = null) => ({ code, severity, message, action, detail });

export async function probe({ budgetMs = 4000, testWidth = 512, testHeight = 512 } = {}) {
  const started = now();
  const reasons = [];
  let gpu = null;
  let encoder = null;
  const storage = { opfs: false, cacheApi: false };

  const deadline = new Promise((res) => setTimeout(() => res('timeout'), budgetMs));
  const work = (async () => {
    if (typeof isSecureContext !== 'undefined' && !isSecureContext) {
      reasons.push(reason(CODES.E_INSECURE_CONTEXT, SEVERITY.FATAL,
        'This page is not running in a secure context.',
        'Open it over https, or http://localhost / http://127.0.0.1 in development.'));
      return;
    }
    if (typeof WebAssembly === 'undefined') {
      reasons.push(reason(CODES.E_NO_WASM, SEVERITY.FATAL, 'WebAssembly is unavailable.', 'Use a current desktop Chrome or Edge.'));
    }
    gpu = await probeGpu(reasons);
    encoder = await probeEncoder(reasons, testWidth, testHeight);
    storage.opfs = Boolean(navigator.storage?.getDirectory);
    storage.cacheApi = typeof caches !== 'undefined';
    if (!storage.opfs && !storage.cacheApi) {
      reasons.push(reason(CODES.E_STORAGE_UNAVAILABLE, SEVERITY.DEGRADE,
        'No persistent storage — model weights will re-download every session.',
        'Leave private browsing, or allow site data for this origin.'));
    }
  })();

  const outcome = await Promise.race([work.then(() => 'done').catch((e) => e), deadline]);
  if (outcome === 'timeout') {
    reasons.push(reason(CODES.E_PROBE_TIMEOUT, SEVERITY.FATAL,
      `Capability check did not finish within ${budgetMs} ms.`,
      'Usually a stalled GPU driver. Restart the browser; if it persists, update your graphics driver.'));
  } else if (outcome !== 'done') {
    reasons.push(reason(CODES.E_INTERNAL, SEVERITY.FATAL, 'The capability check failed unexpectedly.',
      'Reload the page.', { cause: String(outcome?.message ?? outcome) }));
  }

  const fatal = reasons.some((r) => r.severity === SEVERITY.FATAL);
  const degraded = reasons.some((r) => r.severity === SEVERITY.DEGRADE);
  return {
    tier: fatal ? TIER.NONE : degraded ? TIER.STANDARD : TIER.FULL,
    fatal, reasons, gpu, encoder, storage,
    durationMs: Math.round(now() - started),
  };
}

async function probeGpu(reasons) {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    reasons.push(reason(CODES.E_NO_WEBGPU, SEVERITY.FATAL, 'WebGPU is unavailable in this browser.',
      'VJ_VID needs WebGPU. Use desktop Chrome or Edge 121+. On Linux, enable Vulkan in chrome://flags.'));
    return null;
  }
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (err) {
    reasons.push(reason(CODES.E_NO_ADAPTER, SEVERITY.FATAL, 'Requesting a GPU adapter threw.',
      'Update your graphics driver and restart the browser.', { cause: String(err?.message ?? err) }));
    return null;
  }
  if (!adapter) {
    reasons.push(reason(CODES.E_NO_ADAPTER, SEVERITY.FATAL, 'No GPU adapter is available.',
      'Your GPU may be blocklisted. Check chrome://gpu.'));
    return null;
  }
  let device = null;
  try {
    device = await adapter.requestDevice();
  } catch (err) {
    reasons.push(reason(CODES.E_NO_DEVICE, SEVERITY.FATAL, 'Could not acquire a GPU device.',
      'Restart the browser; update your driver.', { cause: String(err?.message ?? err) }));
    return null;
  }
  const hasF16 = adapter.features?.has?.('shader-f16') ?? false;
  const info = {
    hasF16,
    maxBufferSize: device.limits?.maxBufferSize ?? null,
    vendor: adapter.info?.vendor ?? null,
    architecture: adapter.info?.architecture ?? null,
  };
  try { device.destroy?.(); } catch { /* ignore */ }
  return info;
}

async function probeEncoder(reasons, width, height) {
  if (typeof VideoEncoder === 'undefined') {
    reasons.push(reason(CODES.E_NO_WEBCODECS, SEVERITY.FATAL, 'WebCodecs video encoding is unavailable.',
      'Use desktop Chrome or Edge — the video is encoded on-device with WebCodecs.'));
    return null;
  }
  const config = { codec: 'avc1.42001f', width, height, bitrate: 4_000_000, framerate: 30 };
  try {
    const support = await VideoEncoder.isConfigSupported(config);
    if (!support.supported) {
      reasons.push(reason(CODES.E_NO_WEBCODECS, SEVERITY.DEGRADE, 'H.264 hardware encode not confirmed.',
        'Encoding may be slower (software path).', { config }));
      return { hardware: false, codec: config.codec };
    }
    return { hardware: true, codec: config.codec };
  } catch (err) {
    reasons.push(reason(CODES.E_NO_WEBCODECS, SEVERITY.DEGRADE, 'Could not verify the video encoder.',
      'Encoding may fail on this machine.', { cause: String(err?.message ?? err) }));
    return null;
  }
}
