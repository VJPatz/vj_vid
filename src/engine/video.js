/**
 * Video assembly: K keyframe stills -> mp4 (up to 20s), encoded on-device with
 * WebCodecs. Motion comes from RIFE v4.9 (rife.js) — real optical-flow frame
 * interpolation. Each inter-keyframe gap is recursively bisected via RIFE
 * (t=0.5, then 0.5 of each half, etc.) since a single call only gives the
 * exact midpoint, not an arbitrary t — the standard RIFE multi-frame recipe.
 * RIFE won't invent motion the keyframes don't already imply.
 */

import { CODES, SMError, asSMError, throwIfAborted } from '../core/errors.js';
import { LIMITS, clampDuration } from './registry.js';

/**
 * Recursively interpolate `count` evenly-spaced frames between rgbaA (t=0) and
 * rgbaB (t=1). RIFE only gives the exact midpoint of a pair, so arbitrary t
 * means bisecting: interpolate(A,B)->mid(0.5), interpolate(A,mid)->quarter(0.25).
 * Builds a balanced binary tree rather than solving for arbitrary counts.
 */
export async function interpolateBetween(rife, session, rgbaA, rgbaB, count, width, height, onFrame) {
  if (count <= 0) return [];
  // memoized by [lo,hi] frame slot so overlapping recursive calls reuse work
  const slots = new Array(count + 2); // index 0..count+1, endpoints at 0 and count+1
  slots[0] = rgbaA;
  slots[count + 1] = rgbaB;
  let done = 0;

  async function fill(lo, hi) {
    if (hi - lo <= 1) return; // no slot strictly between lo and hi
    const mid = Math.floor((lo + hi) / 2);
    if (!slots[mid]) {
      slots[mid] = await rife.interpolate(session, slots[lo], slots[hi], 0.5, width, height);
      done += 1;
      onFrame?.(done);
    }
    await fill(lo, mid);
    await fill(mid, hi);
  }
  await fill(0, count + 1);

  return slots.slice(1, count + 1);
}

/** Encode keyframes into an mp4 Blob, with RIFE filling motion between them. */
export async function encodeVideo({ keyframes, rife, rifeSession, seconds, fps = LIMITS.FPS, width = LIMITS.WIDTH, height = LIMITS.HEIGHT, onProgress = null, signal = null }) {
  if (typeof VideoEncoder === 'undefined') {
    throw new SMError(CODES.E_NO_WEBCODECS, 'WebCodecs is unavailable — cannot encode video.', { action: 'Use desktop Chrome or Edge.' });
  }
  if (!keyframes?.length) throw new SMError(CODES.E_INTERNAL, 'No keyframes to encode.', { action: 'Try generating again.' });
  if (!rife || !rifeSession) throw new SMError(CODES.E_INTERNAL, 'RIFE model was not loaded.', { action: 'Reload and try again.' });

  const dur = clampDuration(seconds);
  const totalFrames = dur * fps;
  const K = keyframes.length;

  // Load the muxer lazily so it isn't in the initial bundle.
  let Muxer, ArrayBufferTarget;
  try {
    const mod = await import('mp4-muxer');
    Muxer = mod.Muxer;
    ArrayBufferTarget = mod.ArrayBufferTarget;
  } catch (err) {
    throw asSMError(err, CODES.E_ENCODE_FAILED, 'Could not load the mp4 muxer.', { action: 'Run `npm install` and reload.' });
  }

  // Resample keyframes to target resolution if needed.
  const rgbaFrames = [];
  for (const kf of keyframes) rgbaFrames.push(await toRgba(kf, width, height));

  // Distribute totalFrames across K-1 segments; trimmed to totalFrames at the end.
  const segments = Math.max(1, K - 1);
  const framesPerSegment = Math.max(1, Math.round(totalFrames / segments));

  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d', { alpha: false });

  const target = new ArrayBufferTarget();
  const muxer = new Muxer({
    target,
    video: { codec: 'avc', width, height },
    fastStart: 'in-memory',
    firstTimestampBehavior: 'offset',
  });

  const encoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: (e) => { encoderError = e; },
  });
  let encoderError = null;
  encoder.configure({ codec: 'avc1.42001f', width, height, bitrate: 6_000_000, framerate: fps });

  const frameDurUs = Math.round(1_000_000 / fps);
  let encoded = 0;
  // RIFE interpolation dominates cost, so progress is weighted toward it.
  const totalInterpolations = segments * (framesPerSegment - 1 > 0 ? framesPerSegment - 1 : 0);
  let interpolated = 0;

  const encodeOne = async (rgba) => {
    throwIfAborted(signal);
    if (encoderError) throw encoderError;
    const img = new ImageData(rgba, width, height);
    ctx.putImageData(img, 0, 0);
    const frame = new VideoFrame(canvas, { timestamp: encoded * frameDurUs, duration: frameDurUs });
    encoder.encode(frame, { keyFrame: encoded % fps === 0 });
    frame.close();
    encoded += 1;
    if (encoded >= totalFrames) return;
    if (encoded % fps === 0) await new Promise((r) => setTimeout(r, 0));
  };

  try {
    // K=1: nothing to interpolate — hold the single keyframe for the whole clip.
    if (K === 1) {
      for (let f = 0; f < totalFrames; f += 1) await encodeOne(rgbaFrames[0]);
    } else {
      for (let s = 0; s < segments; s += 1) {
        throwIfAborted(signal);
        const a = rgbaFrames[s];
        const b = rgbaFrames[s + 1];
        const between = await interpolateBetween(
          rife, rifeSession, a, b, Math.max(0, framesPerSegment - 1), width, height,
          () => { interpolated += 1; onProgress?.({ frame: encoded, total: totalFrames, fraction: 0.6 * (interpolated / Math.max(1, totalInterpolations)) }); },
        );
        await encodeOne(a);
        for (const mid of between) await encodeOne(mid);
        if (encoded >= totalFrames) break;
      }
      // Last keyframe + pad/trim to exactly totalFrames.
      while (encoded < totalFrames) await encodeOne(rgbaFrames[K - 1]);
    }
    await encoder.flush();
    if (encoderError) throw encoderError;
    muxer.finalize();
  } catch (err) {
    try { encoder.close(); } catch { /* ignore */ }
    throw asSMError(err, CODES.E_ENCODE_FAILED, 'Video encoding failed.', { action: 'Try a shorter clip, or reload and retry.' });
  }
  try { encoder.close(); } catch { /* ignore */ }
  onProgress?.({ frame: totalFrames, total: totalFrames, fraction: 1 });

  return new Blob([target.buffer], { type: 'video/mp4' });
}

/** kf (ImageData-like RGBA) -> RGBA at exactly width x height, resampled via canvas if needed. */
export async function toRgba(kf, width, height) {
  const w = kf.width ?? width;
  const h = kf.height ?? height;
  const data = kf.data instanceof Uint8ClampedArray ? kf.data : new Uint8ClampedArray(kf.data);
  if (w === width && h === height) return data;
  const src = new OffscreenCanvas(w, h);
  src.getContext('2d').putImageData(new ImageData(data, w, h), 0, 0);
  const dst = new OffscreenCanvas(width, height);
  const dctx = dst.getContext('2d');
  dctx.drawImage(src, 0, 0, width, height);
  return dctx.getImageData(0, 0, width, height).data;
}
