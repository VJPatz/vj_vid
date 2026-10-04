/**
 * Pinned model registry — nothing else in the app constructs a model URL.
 *
 * SD-Turbo (SD2.1-distilled, one-step): OpenCLIP-ViT-H encoder (hidden_size
 * 1024, 77 tokens), fp32 graph I/O (verified via onnx.load() dtype dump —
 * the model card's "fp16" refers to internal weight storage only), VAE scale
 * 0.18215, 4ch 64x64 latents for 512x512. Repo must support non-CUDA EPs —
 * tlwu/sd-turbo-onnxruntime does not; schmuell/sd-turbo-ort-web does
 * (same repo Microsoft's own browser SD-Turbo sample uses). Tokenizer:
 * openai/clip-vit-large-patch14 (byte-identical BPE vocab; sd-turbo's own
 * tokenizer/ is a legacy format transformers.js can't load). onnxruntime-web
 * pinned to exact 1.17.1 — 1.20.0's WebGPU EP throws a float16 dtype error
 * against this repo; re-verify end-to-end before bumping.
 *
 * Every file lists candidate paths tried in order — community ONNX repos
 * get reorganized, and a hard-coded path is the easiest way this breaks later.
 */

export const MB = 1024 * 1024;

const SDTURBO_BASE_URL = 'https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main';

export const SDTURBO = Object.freeze({
  id: 'sd-turbo',
  label: 'SD-Turbo',
  license: 'Stability AI Non-Commercial Research Community',
  licenseUrl: 'https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE',
  modelUrl: SDTURBO_BASE_URL,
  baseUrl: SDTURBO_BASE_URL,
  tokenizerRepo: 'openai/clip-vit-large-patch14',

  files: Object.freeze({
    textEncoder: Object.freeze({
      candidates: Object.freeze(['text_encoder/model.onnx']),
      approxBytes: 681 * MB,
      required: true,
    }),
    unet: Object.freeze({
      candidates: Object.freeze(['unet/model.onnx']),
      approxBytes: 1730 * MB,
      required: true,
    }),
    vaeDecoder: Object.freeze({
      candidates: Object.freeze(['vae_decoder/model.onnx']),
      approxBytes: 99 * MB,
      required: true,
    }),
  }),

  residentBytes: 2600 * MB,

  pipeline: Object.freeze({
    imageSize: 512,
    latentSize: 64,
    latentChannels: 4,
    maxTokens: 77,
    hiddenSize: 1024,
    vaeScaleFactor: 0.18215,
    timestep: 999, // single-step native regime
    trainSteps: 1000,
    betaStart: 0.00085,
    betaEnd: 0.012,
  }),
});

/**
 * RIFE v4.9 (optical-flow frame interpolation between two keyframes).
 * edgetools/rife chosen because it targets onnxruntime-web (WASM/WebGPU) —
 * most RIFE ONNX exports are CUDA/DirectML desktop-only.
 * I/O: planar RGB (CHW), 0..1, dims a multiple of 32 (512 already satisfies this).
 */
export const RIFE = Object.freeze({
  id: 'rife',
  label: 'RIFE v4.9',
  license: 'MIT',
  licenseUrl: 'https://huggingface.co/edgetools/rife/blob/main/LICENSE',
  modelUrl: 'https://huggingface.co/edgetools/rife',
  baseUrl: 'https://huggingface.co/edgetools/rife/resolve/main',
  files: Object.freeze({
    model: Object.freeze({
      candidates: Object.freeze(['rife49.onnx']),
      approxBytes: 22 * MB,
      required: true,
    }),
  }),
  residentBytes: 60 * MB, // small graph, generous headroom over the 22MB weights
  imageSize: 512, // must match SDTURBO.pipeline.imageSize — no cross-model resize step exists
});

export function rifeFileList() {
  const base = RIFE.baseUrl.replace(/\/+$/, '');
  return Object.entries(RIFE.files).map(([name, spec]) => {
    const urls = spec.candidates.map((p) => `${base}/${p.replace(/^\//, '')}`);
    return { name, url: urls[0], urls, key: `${RIFE.id}/${name}`, approxBytes: spec.approxBytes, required: spec.required };
  });
}

export function rifeDownloadBytes() {
  return rifeFileList().reduce((sum, f) => sum + f.approxBytes, 0);
}

/**
 * Motion parameters — how K independent keyframes read as one moving scene.
 * ANCHOR_MODE: 'chain' blends each frame toward the PREVIOUS frame's x0
 *   (frame0->frame1->frame2->...) — real progressive motion, can drift over
 *   a long clip since deviations compound each step. 'first' blends every
 *   frame toward frame 0 (hub-and-spoke) — strong identity lock, but every
 *   frame reads as "frame 0 with distortion" rather than motion.
 * ANCHOR_STRENGTH: pulls each keyframe's x0 toward its anchor (0 = unrelated
 *   images, 1 = copies). Too high (>0.5) and frames barely differ — RIFE then
 *   interpolates near-duplicates, which reads as static distortion, not a
 *   RIFE failure. In 'chain' mode, keep this moderate — high values still
 *   compound drift over many frames even though each individual step looks fine.
 * LATENT_DRIFT: how far the shared base noise translates across the clip, in
 *   latent cells — reads as camera movement. Needs to be large enough that
 *   its effect on the final image survives ANCHOR_STRENGTH's pull back.
 * SECONDS_PER_KEYFRAME: below ~1s diffusion cost dominates; above ~2s it
 *   reads as a slideshow.
 * ANCHOR_STRENGTH/LATENT_DRIFT are also eased linearly across the clip from
 *   *_START (t=0) to *_END (t=1) — e.g. loosen anchor over time so a long
 *   clip can drift further by the end than the start. Flat by default
 *   (START === END == the scalar above); set them apart to ramp.
 */
export const MOTION = Object.freeze({
  SECONDS_PER_KEYFRAME: 1.5,
  MIN_KEYFRAMES: 2,
  MAX_KEYFRAMES: 14,
  ANCHOR_MODE: 'chain',
  ANCHOR_STRENGTH: 0.35,
  LATENT_DRIFT: 6,
  // Default ramp: strong identity lock at the start, loosening as the clip
  // progresses, so later frames can move more. Override via URL params
  // (?anchorStart=&anchorEnd=&driftStart=&driftEnd=) if you want to recalibrate.
  ANCHOR_STRENGTH_START: 0.45,
  ANCHOR_STRENGTH_END: 0.2,
  LATENT_DRIFT_START: 3,
  LATENT_DRIFT_END: 10,
});

/** Hard product limits. The 20s cap is a deliberate scope decision (lightweight/fast). */
export const LIMITS = Object.freeze({
  MIN_SECONDS: 2,
  MAX_SECONDS: 20,
  DEFAULT_SECONDS: 8,
  FPS: 30,
  WIDTH: 512,
  HEIGHT: 512,
});

/** Clamp a requested duration into the supported range. */
export function clampDuration(seconds) {
  const s = Number(seconds);
  if (!Number.isFinite(s)) return LIMITS.DEFAULT_SECONDS;
  return Math.max(LIMITS.MIN_SECONDS, Math.min(LIMITS.MAX_SECONDS, Math.round(s)));
}

/** How many diffusion keyframes a clip of this length gets. Pure — UI shows it pre-run. */
export function keyframeCountFor(durationSeconds, { secondsPer = MOTION.SECONDS_PER_KEYFRAME } = {}) {
  const seconds = clampDuration(durationSeconds);
  const raw = Math.round(seconds / Math.max(0.5, secondsPer));
  return Math.max(MOTION.MIN_KEYFRAMES, Math.min(MOTION.MAX_KEYFRAMES, raw));
}

export function totalRenderFrames(durationSeconds, fps = LIMITS.FPS) {
  return clampDuration(durationSeconds) * fps;
}

/** Effective file list — each entry's `urls` are tried in order, first to answer wins. */
export function sdTurboFileList() {
  const base = SDTURBO.baseUrl.replace(/\/+$/, '');
  return Object.entries(SDTURBO.files).map(([name, spec]) => {
    const urls = spec.candidates.map((p) => `${base}/${p.replace(/^\//, '')}`);
    return { name, url: urls[0], urls, key: `${SDTURBO.id}/${name}`, approxBytes: spec.approxBytes, required: spec.required };
  });
}

export function totalDownloadBytes() {
  return sdTurboFileList().reduce((sum, f) => sum + f.approxBytes, 0);
}
