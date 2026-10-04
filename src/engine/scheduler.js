/**
 * Diffusion scheduler math for SD-Turbo (scaled_linear, epsilon-prediction,
 * single-step at t=999). Pure/dependency-free — unit-tested without a GPU.
 */

export const DEFAULTS = Object.freeze({
  betaStart: 0.00085,
  betaEnd: 0.012,
  trainSteps: 1000,
});

/** Cumulative product of alphas for the scaled_linear schedule. */
export function scaledLinearAlphasCumprod({ betaStart, betaEnd, trainSteps } = DEFAULTS) {
  betaStart ??= DEFAULTS.betaStart;
  betaEnd ??= DEFAULTS.betaEnd;
  trainSteps ??= DEFAULTS.trainSteps;
  if (!Number.isInteger(trainSteps) || trainSteps < 2) {
    throw new RangeError(`trainSteps must be an integer >= 2, got ${trainSteps}`);
  }
  const lo = Math.sqrt(betaStart);
  const hi = Math.sqrt(betaEnd);
  const out = new Float64Array(trainSteps);
  let acc = 1;
  for (let i = 0; i < trainSteps; i += 1) {
    const t = i / (trainSteps - 1);
    const sqrtBeta = lo + t * (hi - lo);
    const beta = sqrtBeta * sqrtBeta;
    acc *= 1 - beta;
    out[i] = acc;
  }
  return out;
}

/** Initial latent scale for DDPM-style schedules (SD2.1 family) — always 1.0. */
export function initNoiseSigma() {
  return 1.0;
}

/** One epsilon-prediction step -> x0. Writes into `out` if given (may alias `latents`). */
export function epsilonToSample(latents, noisePred, alphaProdT, out = null) {
  if (latents.length !== noisePred.length) {
    throw new RangeError(`latents (${latents.length}) and noisePred (${noisePred.length}) length mismatch`);
  }
  if (!(alphaProdT > 0) || alphaProdT > 1) {
    throw new RangeError(`alphaProdT must be in (0, 1], got ${alphaProdT}`);
  }
  const target = out ?? new Float32Array(latents.length);
  const sqrtAlpha = Math.sqrt(alphaProdT);
  const sqrtOneMinus = Math.sqrt(1 - alphaProdT);
  for (let i = 0; i < latents.length; i += 1) {
    target[i] = (latents[i] - sqrtOneMinus * noisePred[i]) / sqrtAlpha;
  }
  return target;
}

/** Forward diffusion: q(x_t | x_0) = sqrt(a_t) x0 + sqrt(1 - a_t) noise. */
export function addNoise(x0, noise, alphaProdT, out = null) {
  const target = out ?? new Float32Array(x0.length);
  const sqrtAlpha = Math.sqrt(alphaProdT);
  const sqrtOneMinus = Math.sqrt(1 - alphaProdT);
  for (let i = 0; i < x0.length; i += 1) {
    target[i] = sqrtAlpha * x0[i] + sqrtOneMinus * noise[i];
  }
  return target;
}

/** Scale latents for the VAE decoder: diffusers divides by scaling_factor. */
export function scaleLatentsForDecode(latents, vaeScaleFactor = 0.18215, out = null) {
  const target = out ?? new Float32Array(latents.length);
  const inv = 1 / vaeScaleFactor;
  for (let i = 0; i < latents.length; i += 1) target[i] = latents[i] * inv;
  return target;
}

/** NCHW float tensor (N=1, C=3, ~[-1,1]) -> packed RGBA8 for ImageData. */
export function nchwToRgba(tensor, width, height) {
  const pixels = width * height;
  if (tensor.length < pixels * 3) {
    throw new RangeError(`tensor too small: need ${pixels * 3} for ${width}x${height}, got ${tensor.length}`);
  }
  const out = new Uint8ClampedArray(pixels * 4);
  const g = pixels;
  const b = pixels * 2;
  for (let i = 0; i < pixels; i += 1) {
    const o = i * 4;
    out[o] = (tensor[i] * 0.5 + 0.5) * 255;
    out[o + 1] = (tensor[g + i] * 0.5 + 0.5) * 255;
    out[o + 2] = (tensor[b + i] * 0.5 + 0.5) * 255;
    out[o + 3] = 255;
  }
  return out;
}
