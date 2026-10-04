/**
 * RIFE v4.9 wrapper — optical-flow frame interpolation between two keyframes.
 * Same load/use/release-in-finally discipline as sdturbo.js, kept consistent
 * even though this graph (~22MB, ~60MB resident) isn't the memory pressure point.
 *
 * I/O (NOT scheduler.js's [-1,1] VAE convention):
 *   img0, img1: planar RGB (CHW), float32, 0..1
 *   timestep:   float32 [1], 0 = img0, 1 = img1
 *   output:     planar RGB (CHW), float32, 0..1
 * Dims must be a multiple of 32 — 512x512 satisfies this with no padding.
 */

import { createSession, releaseSession } from './ort.js';
import { RIFE, rifeFileList } from './registry.js';
import { CODES, SMError } from '../core/errors.js';
import { logger } from '../core/log.js';

export class Rife {
  constructor(cache) {
    this.cache = cache;
    this.bytes = null;
    this._log = logger.child('rife');
  }

  async fetchAll({ onProgress = null, signal = null } = {}) {
    const [file] = rifeFileList();
    this.bytes = await this.cache.fetchWithCache({
      urls: file.urls, key: file.key, approxBytes: file.approxBytes, signal,
      onProgress: onProgress ? (p) => onProgress({ ...p, file: file.name }) : null,
    });
  }

  async with(fn) {
    if (!this.bytes) throw new SMError(CODES.E_INTERNAL, 'RIFE model was not loaded.', { action: 'Reload and try again.' });
    const { session } = await createSession({ bytes: this.bytes, label: 'rife', preferWebgpu: true, freeDimensionOverrides: {} });
    try {
      return await fn(session);
    } finally {
      await releaseSession(session, 'rife');
    }
  }

  /** RGBA8 (0..255) -> planar RGB float32 (0..1). Drops alpha. */
  static rgbaToPlanarRgb01(rgba, width, height) {
    const pixels = width * height;
    const out = new Float32Array(pixels * 3);
    const g = pixels;
    const b = pixels * 2;
    for (let i = 0; i < pixels; i += 1) {
      const o = i * 4;
      out[i] = rgba[o] / 255;
      out[g + i] = rgba[o + 1] / 255;
      out[b + i] = rgba[o + 2] / 255;
    }
    return out;
  }

  /** Planar RGB float32 (0..1) -> RGBA8 (0..255, alpha=255). */
  static planarRgb01ToRgba(planar, width, height) {
    const pixels = width * height;
    const out = new Uint8ClampedArray(pixels * 4);
    const g = pixels;
    const b = pixels * 2;
    for (let i = 0; i < pixels; i += 1) {
      const o = i * 4;
      out[o] = planar[i] * 255;
      out[o + 1] = planar[g + i] * 255;
      out[o + 2] = planar[b + i] * 255;
      out[o + 3] = 255;
    }
    return out;
  }

  /** Interpolate one frame between two RGBA keyframes at timestep t (0=A, 1=B). */
  async interpolate(session, rgbaA, rgbaB, t, width = RIFE.imageSize, height = RIFE.imageSize) {
    const ort = await import('onnxruntime-web/webgpu');
    const img0 = new ort.Tensor('float32', Rife.rgbaToPlanarRgb01(rgbaA, width, height), [1, 3, height, width]);
    const img1 = new ort.Tensor('float32', Rife.rgbaToPlanarRgb01(rgbaB, width, height), [1, 3, height, width]);
    const timestep = new ort.Tensor('float32', Float32Array.of(t), [1]);
    const out = await session.run({ img0, img1, timestep });
    const key = session.outputNames?.[0] ?? 'output'; // output name not standardized across exports
    return Rife.planarRgb01ToRgba(out[key].data, width, height);
  }
}
