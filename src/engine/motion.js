/**
 * Latent motion: how K independently-sampled keyframes read as one moving
 * scene instead of a slideshow. Pure, unit-tested, no GPU.
 *
 */

import { gaussianNoise } from '../core/rng.js';

/** Shared base noise field for a clip. */
export function baseNoise({ seed, latentChannels, latentSize }) {
  return gaussianNoise(latentChannels * latentSize * latentSize, seed);
}

/** Translate the base noise field by a sub-pixel offset (bilinear, wrap-around), 0 at t=0 to driftCells at t=1, diagonal pan. */
export function driftedNoise(base, { latentChannels, latentSize, driftCells, t }) {
  const dx = driftCells * t;
  const dy = driftCells * t * 0.6; // shallower vertical component -> diagonal drift
  const out = new Float32Array(base.length);
  const S = latentSize;
  for (let c = 0; c < latentChannels; c += 1) {
    const plane = c * S * S;
    for (let y = 0; y < S; y += 1) {
      for (let x = 0; x < S; x += 1) {
        const sx = x + dx;
        const sy = y + dy;
        out[plane + y * S + x] = bilinearWrap(base, plane, S, sx, sy);
      }
    }
  }
  return out;
}

function bilinearWrap(field, plane, S, fx, fy) {
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const tx = fx - x0;
  const ty = fy - y0;
  const wx0 = ((x0 % S) + S) % S;
  const wx1 = ((x0 + 1) % S + S) % S;
  const wy0 = ((y0 % S) + S) % S;
  const wy1 = ((y0 + 1) % S + S) % S;
  const a = field[plane + wy0 * S + wx0];
  const b = field[plane + wy0 * S + wx1];
  const c = field[plane + wy1 * S + wx0];
  const d = field[plane + wy1 * S + wx1];
  const top = a + (b - a) * tx;
  const bot = c + (d - c) * tx;
  return top + (bot - top) * ty;
}

/** Pull a predicted x0 latent toward an anchor x0 by `strength` (0..1). */
export function anchorBlend(x0, anchorX0, strength, out = null) {
  const target = out ?? new Float32Array(x0.length);
  const s = Math.min(1, Math.max(0, strength));
  for (let i = 0; i < x0.length; i += 1) target[i] = x0[i] * (1 - s) + anchorX0[i] * s;
  return target;
}

/** Spherical interpolation between two flattened embeddings; falls back to lerp near-colinear. */
export function slerp(a, b, t, out = null) {
  const target = out ?? new Float32Array(a.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i += 1) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const denom = Math.sqrt(na * nb) || 1e-8;
  let cos = dot / denom;
  cos = Math.min(1, Math.max(-1, cos));
  if (cos > 0.9995) {
    for (let i = 0; i < a.length; i += 1) target[i] = a[i] + (b[i] - a[i]) * t;
    return target;
  }
  const theta = Math.acos(cos);
  const sinTheta = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / sinTheta;
  const wb = Math.sin(t * theta) / sinTheta;
  for (let i = 0; i < a.length; i += 1) target[i] = wa * a[i] + wb * b[i];
  return target;
}

/** Linear ease between two values over normalized position t (0..1). */
export function lerpValue(start, end, t) {
  return start + (end - start) * Math.min(1, Math.max(0, t));
}

/** Per-keyframe plan: normalized position t + which prompt-embedding pair to blend. */
export function keyframePlan(count, prompts = 1) {
  const out = [];
  const segs = Math.max(1, prompts - 1);
  for (let i = 0; i < count; i += 1) {
    const t = count <= 1 ? 0 : i / (count - 1);
    const p = t * segs;
    const embedA = Math.min(prompts - 1, Math.floor(p));
    const embedB = Math.min(prompts - 1, embedA + 1);
    out.push({ index: i, t, embedA, embedB, embedT: p - embedA });
  }
  return out;
}
