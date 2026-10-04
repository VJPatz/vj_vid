/**
 * The orchestrator: prompt -> video, entirely on-device.
 * PROBE -> DOWNLOAD -> ENCODE_TEXT -> KEYFRAMES -> DECODE -> RIFE_DOWNLOAD -> ENCODE_VIDEO.
 *
 * Each model (text encoder, UNet, VAE decoder, RIFE) is loaded for its phase
 * and released before the next, so peak RAM stays near the largest single
 * model. Determinism: (prompt, seed, seconds) always reproduces the same video.
 *
 * Keyframes are independently sampled from a shared, slowly-drifting base
 * noise (motion.js) — no img2img chaining, so no drift/collapse. Motion
 * between keyframes comes from RIFE interpolation (video.js), not a camera effect.
 */

import { Emitter } from '../core/events.js';
import { logger } from '../core/log.js';
import { CODES, SMError, isCancel, throwIfAborted } from '../core/errors.js';
import { hashSeed } from '../core/rng.js';
import { WeightCache } from './cache.js';
import { SDTurbo } from './sdturbo.js';
import { Rife } from './rife.js';
import { tokenize } from './tokenizer.js';
import { SDTURBO, LIMITS, MOTION, clampDuration, keyframeCountFor, sdTurboFileList, rifeFileList } from './registry.js';
import { initNoiseSigma, epsilonToSample, scaledLinearAlphasCumprod } from './scheduler.js';
import { baseNoise, driftedNoise, anchorBlend, slerp, keyframePlan, lerpValue } from './motion.js';
import { encodeVideo } from './video.js';
import { buildPromptSchedule } from './prompt.js';

export { buildPromptSchedule };

export const PHASE = Object.freeze({
  IDLE: 'idle', PROBE: 'probe', DOWNLOAD: 'download', ENCODE_TEXT: 'encode_text',
  KEYFRAMES: 'keyframes', DECODE: 'decode', RIFE_DOWNLOAD: 'rife_download', ENCODE_VIDEO: 'encode_video',
  DONE: 'done', FAILED: 'failed', CANCELLED: 'cancelled',
});

// Relative weights for the overall progress bar. Download dominates a cold
// run; on a warm run it collapses and the rest renormalize.
const WEIGHTS = { download: 0.45, encode_text: 0.03, keyframes: 0.2, decode: 0.05, rife_download: 0.05, encode_video: 0.22 };

export class Pipeline {
  constructor() {
    this.events = new Emitter();
    this._log = logger.child('pipeline');
    this._abort = null;
    this.phase = PHASE.IDLE;
  }

  on(type, fn) { return this.events.on(type, fn); }
  cancel() { this._abort?.abort(); }

  _setPhase(phase, extra = {}) {
    this.phase = phase;
    this.events.emit('phase', { phase, ...extra });
  }

  // SD-Turbo and RIFE cache independently, so each gets its own warm flag.
  _overall(phase, fractionWithinPhase, warm, rifeWarm = warm) {
    const w = { ...WEIGHTS };
    if (warm) w.download = 0.02;
    if (rifeWarm) w.rife_download = 0.01;
    const total = Object.values(w).reduce((a, b) => a + b, 0);
    const order = ['download', 'encode_text', 'keyframes', 'decode', 'rife_download', 'encode_video'];
    let acc = 0;
    for (const k of order) {
      if (k === phase) { acc += w[k] * Math.min(1, Math.max(0, fractionWithinPhase)); break; }
      acc += w[k];
    }
    this.events.emit('progress', { overall: Math.min(1, acc / total), phase, fraction: fractionWithinPhase });
  }

  /** Image mode: one prompt -> one image. No keyframe loop, no video encode. */
  async generateImage({ prompt, seed = null } = {}) {
    this._abort = new AbortController();
    const signal = this._abort.signal;
    const seedNum = seed == null || seed === '' ? (Math.random() * 2 ** 32) >>> 0 : (typeof seed === 'number' ? seed >>> 0 : hashSeed(seed));
    const P = SDTURBO.pipeline;
    const started = Date.now();

    try {
      throwIfAborted(signal);

      const cache = await WeightCache.open();
      await cache.requestPersistence();
      const fileList = sdTurboFileList();
      const warm = await cache.hasAll(fileList.map((f) => f.key));
      this._setPhase(PHASE.DOWNLOAD, { warm });
      const model = new SDTurbo(cache);
      const fileBytes = {};
      await model.fetchAll({
        signal,
        onProgress: (p) => {
          fileBytes[p.file] = { loaded: p.loaded, total: p.total ?? p.loaded };
          const loaded = Object.values(fileBytes).reduce((a, b) => a + b.loaded, 0);
          const total = Object.values(fileBytes).reduce((a, b) => a + b.total, 0) || 1;
          this.events.emit('download', { ...p, loadedAll: loaded, totalAll: total, warm });
          this._overall('download', loaded / total, warm);
        },
      });

      this._setPhase(PHASE.ENCODE_TEXT);
      const toks = await tokenize(buildPromptSchedule(prompt)[0]);
      const hidden = await model.withTextEncoder(async (session) => model.encodeText(session, toks));
      this._overall('encode_text', 1, false);

      this._setPhase(PHASE.KEYFRAMES, { keyframes: 1 });
      const base = baseNoise({ seed: seedNum, latentChannels: P.latentChannels, latentSize: P.latentSize });
      const alphas = scaledLinearAlphasCumprod({ betaStart: P.betaStart, betaEnd: P.betaEnd, trainSteps: P.trainSteps });
      const alphaT = alphas[P.timestep];
      const sigma0 = initNoiseSigma();
      const noise = sigma0 === 1 ? base : base.map((v) => v * sigma0);

      const latent = await model.withUnet(async (session) => {
        const eps = await model.runUnet(session, { sample: noise, timestep: P.timestep, hidden });
        this._overall('keyframes', 1, false);
        this.events.emit('keyframe', { index: 0, count: 1 });
        return epsilonToSample(noise, eps, alphaT);
      });

      this._setPhase(PHASE.DECODE);
      const rgba = await model.withVaeDecoder(async (session) => model.decode(session, latent));
      const image = new ImageData(rgba, P.imageSize, P.imageSize);
      this._overall('decode', 1, false);

      this._setPhase(PHASE.DONE);
      const meta = { prompt, seed: seedNum, ms: Date.now() - started };
      this.events.emit('done', { meta });
      return { image, meta };
    } catch (err) {
      if (isCancel(err)) {
        this._setPhase(PHASE.CANCELLED);
        this.events.emit('cancelled', {});
      } else {
        this._setPhase(PHASE.FAILED);
        const smerr = err instanceof SMError ? err : new SMError(CODES.E_INTERNAL, err?.message ?? String(err), { action: 'Reload and try again.', cause: err });
        this._log.error('image generation failed', { code: smerr.code, message: smerr.message });
        this.events.emit('error', { error: smerr });
      }
      throw err;
    } finally {
      this._abort = null;
    }
  }

  /** Shared first half of generate(): download, text-encode, K-keyframe UNet loop, VAE decode. */
  async _prepareKeyframes({ prompt, seconds, seed, signal, motion = MOTION }) {
    const dur = clampDuration(seconds);
    const seedNum = seed == null || seed === '' ? (Math.random() * 2 ** 32) >>> 0 : (typeof seed === 'number' ? seed >>> 0 : hashSeed(seed));
    const K = keyframeCountFor(dur);
    const P = SDTURBO.pipeline;

    // ---- DOWNLOAD -----------------------------------------------------
    const cache = await WeightCache.open();
    await cache.requestPersistence();
    const fileList = sdTurboFileList();
    const warm = await cache.hasAll(fileList.map((f) => f.key));
    this._setPhase(PHASE.DOWNLOAD, { warm });
    const model = new SDTurbo(cache);
    const fileBytes = {};
    await model.fetchAll({
      signal,
      onProgress: (p) => {
        fileBytes[p.file] = { loaded: p.loaded, total: p.total ?? p.loaded };
        const loaded = Object.values(fileBytes).reduce((a, b) => a + b.loaded, 0);
        const total = Object.values(fileBytes).reduce((a, b) => a + b.total, 0) || 1;
        this.events.emit('download', { ...p, loadedAll: loaded, totalAll: total, warm });
        this._overall('download', loaded / total, warm);
      },
    });

    // ---- ENCODE TEXT ----------------------------------------------------
    this._setPhase(PHASE.ENCODE_TEXT);
    const prompts = buildPromptSchedule(prompt);
    const embeds = await model.withTextEncoder(async (session) => {
      const out = [];
      for (let i = 0; i < prompts.length; i += 1) {
        throwIfAborted(signal);
        const toks = await tokenize(prompts[i]);
        out.push(await model.encodeText(session, toks));
        this._overall('encode_text', (i + 1) / prompts.length, warm);
      }
      return out;
    });
    const hiddenDims = embeds[0].dims;

    // ---- KEYFRAMES (single-step UNet each, shared drifting noise) --------
    this._setPhase(PHASE.KEYFRAMES, { keyframes: K });
    const plan = keyframePlan(K, prompts.length);
    const base = baseNoise({ seed: seedNum, latentChannels: P.latentChannels, latentSize: P.latentSize });
    const alphas = scaledLinearAlphasCumprod({ betaStart: P.betaStart, betaEnd: P.betaEnd, trainSteps: P.trainSteps });
    const alphaT = alphas[P.timestep];
    const sigma0 = initNoiseSigma();

    // Anchor mode: 'first' blends every frame toward frame 0 (hub-and-spoke —
    // strong identity lock, but every frame looks like a variation of frame 0,
    // which reads as "static + distortion" rather than progressive motion).
    // 'chain' blends each frame toward the PREVIOUS frame (a path — frame0 ->
    // frame1 -> frame2 -> ...), which produces real progressive motion but can
    // drift over a long clip since small deviations compound each step. Chain
    // is the default; keep ANCHOR_STRENGTH moderate (not high) to limit drift.
    const latents = await model.withUnet(async (session) => {
      const results = new Array(K);
      let anchorX0 = null; // frame 0's x0 (anchor mode 'first')
      let prevX0 = null;   // previous frame's x0 (anchor mode 'chain')
      for (const step of plan) {
        throwIfAborted(signal);
        // initial latent = drifted shared noise, scaled by init sigma (=1 here)
        const driftCells = lerpValue(
          motion.LATENT_DRIFT_START ?? motion.LATENT_DRIFT,
          motion.LATENT_DRIFT_END ?? motion.LATENT_DRIFT,
          step.t,
        );
        const noise = driftedNoise(base, {
          latentChannels: P.latentChannels, latentSize: P.latentSize, driftCells, t: step.t,
        });
        if (sigma0 !== 1) for (let i = 0; i < noise.length; i += 1) noise[i] *= sigma0;
        // blended prompt embedding for this frame
        const hidden = step.embedA === step.embedB
          ? embeds[step.embedA]
          : { data: slerp(embeds[step.embedA].data, embeds[step.embedB].data, step.embedT), dims: hiddenDims };
        // one UNet step -> eps -> x0
        const eps = await model.runUnet(session, { sample: noise, timestep: P.timestep, hidden });
        let x0 = epsilonToSample(noise, eps, alphaT);
        if (step.index === 0) {
          anchorX0 = x0.slice();
        } else {
          const target = motion.ANCHOR_MODE === 'first' ? anchorX0 : prevX0;
          const strength = lerpValue(
            motion.ANCHOR_STRENGTH_START ?? motion.ANCHOR_STRENGTH,
            motion.ANCHOR_STRENGTH_END ?? motion.ANCHOR_STRENGTH,
            step.t,
          );
          x0 = anchorBlend(x0, target, strength);
        }
        prevX0 = x0.slice();
        results[step.index] = x0;
        this._overall('keyframes', (step.index + 1) / K, warm);
        this.events.emit('keyframe', { index: step.index, count: K });
      }
      return results;
    });

    // ---- DECODE ---------------------------------------------------------
    this._setPhase(PHASE.DECODE);
    let keyframes = await model.withVaeDecoder(async (session) => {
      const imgs = new Array(K);
      for (let i = 0; i < K; i += 1) {
        throwIfAborted(signal);
        const rgba = await model.decode(session, latents[i]);
        imgs[i] = new ImageData(rgba, P.imageSize, P.imageSize);
        this._overall('decode', (i + 1) / K, warm);
      }
      return imgs;
    });

    return { model, keyframes, K, dur, seedNum, warm };
  }

  /** Video mode: prompt -> keyframes -> RIFE-interpolated mp4. seconds clamped to 2..20.
   * `motion` optionally overrides registry.js's MOTION (ANCHOR_STRENGTH/LATENT_DRIFT) for tuning. */
  async generate({ prompt, seconds = LIMITS.DEFAULT_SECONDS, seed = null, motion = MOTION } = {}) {
    this._abort = new AbortController();
    const signal = this._abort.signal;
    const started = Date.now();

    try {
      throwIfAborted(signal);
      const { keyframes, K, dur, seedNum, warm } = await this._prepareKeyframes({ prompt, seconds, seed, signal, motion });

      // ---- RIFE DOWNLOAD (small, but still cache-checked like SD-Turbo) ----
      const rifeCache = await WeightCache.open();
      const rifeFiles = rifeFileList();
      const rifeWarm = await rifeCache.hasAll(rifeFiles.map((f) => f.key));
      this._setPhase(PHASE.RIFE_DOWNLOAD, { warm: rifeWarm });
      const rife = new Rife(rifeCache);
      await rife.fetchAll({
        signal,
        onProgress: (p) => this._overall('rife_download', p.fraction ?? (p.loaded / (p.total || 1)), warm, rifeWarm),
      });

      // ---- ENCODE VIDEO (RIFE interpolation + WebCodecs encode) -----------
      this._setPhase(PHASE.ENCODE_VIDEO);
      const blob = await rife.with(async (rifeSession) => encodeVideo({
        keyframes, rife, rifeSession, seconds: dur, fps: LIMITS.FPS, width: LIMITS.WIDTH, height: LIMITS.HEIGHT, signal,
        onProgress: (p) => this._overall('encode_video', p.fraction, warm, rifeWarm),
      }));

      this._setPhase(PHASE.DONE);
      const meta = { prompt, seconds: dur, seed: seedNum, keyframes: K, fps: LIMITS.FPS, ms: Date.now() - started };
      this.events.emit('done', { blob, meta });
      return { blob, keyframes, meta };
    } catch (err) {
      if (isCancel(err)) {
        this._setPhase(PHASE.CANCELLED);
        this.events.emit('cancelled', {});
      } else {
        this._setPhase(PHASE.FAILED);
        const smerr = err instanceof SMError ? err : new SMError(CODES.E_INTERNAL, err?.message ?? String(err), { action: 'Reload and try again.', cause: err });
        this._log.error('generation failed', { code: smerr.code, message: smerr.message });
        this.events.emit('error', { error: smerr });
      }
      throw err;
    } finally {
      this._abort = null;
    }
  }
}
