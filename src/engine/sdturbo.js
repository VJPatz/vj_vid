/**
 * SD-Turbo model wrapper — text encoder, UNet, VAE decoder. Each phase's
 * session is created, used, then released (with* helpers enforce this in a
 * finally) so peak RAM stays near the UNet alone (~1.9 GB).
 * I/O names vary across ONNX exports, so every run introspects
 * inputNames/outputNames rather than hard-coding them. Token ids try int64,
 * fall back to int32.
 */

import { createSession, releaseSession, makeFloatTensor, loadOrt } from './ort.js';
import { SDTURBO, sdTurboFileList } from './registry.js';
import { nchwToRgba, scaleLatentsForDecode } from './scheduler.js';
import { CODES, SMError } from '../core/errors.js';
import { logger } from '../core/log.js';

const P = SDTURBO.pipeline;

export class SDTurbo {
  constructor(cache) {
    this.cache = cache;
    this.bytes = new Map(); // name -> Uint8Array
    this._log = logger.child('sdturbo');
  }

  /** Download (or load from cache) every required file. onProgress gets per-file bytes. */
  async fetchAll({ onProgress = null, signal = null } = {}) {
    const files = sdTurboFileList();
    for (const f of files) {
      const bytes = await this.cache.fetchWithCache({
        urls: f.urls, key: f.key, approxBytes: f.approxBytes, signal,
        onProgress: onProgress ? (p) => onProgress({ ...p, file: f.name }) : null,
      });
      this.bytes.set(f.name, bytes);
    }
  }

  /* -------- phase-scoped session helpers (release in finally) -------- */

  async _with(name, label, fn, { preferWebgpu = true, freeDimensionOverrides = null } = {}) {
    const bytes = this.bytes.get(name);
    if (!bytes) throw new SMError(CODES.E_INTERNAL, `Model file "${name}" was not loaded.`, { action: 'Reload and try again.' });
    const { session } = await createSession({ bytes, label, preferWebgpu, freeDimensionOverrides });
    try {
      return await fn(session);
    } finally {
      await releaseSession(session, label);
    }
  }

  // Text encoder pinned to WASM — its fused Attention kernel rejects some
  // mask shapes at run time, past where ort.js's fallback can catch it, and
  // it's one pass per prompt so the WebGPU speedup isn't worth the risk.
  // freeDimensionOverrides pin each graph's symbolic dims (required for this
  // repo's exports to select the right kernel variant).
  withTextEncoder(fn) {
    return this._with('textEncoder', 'text_encoder', fn, {
      preferWebgpu: false,
      freeDimensionOverrides: { batch_size: 1 },
    });
  }
  withUnet(fn) {
    return this._with('unet', 'unet', fn, {
      freeDimensionOverrides: {
        batch_size: 1, num_channels: P.latentChannels, height: P.latentSize, width: P.latentSize, sequence_length: P.maxTokens,
      },
    });
  }
  withVaeDecoder(fn) {
    return this._with('vaeDecoder', 'vae_decoder', fn, {
      freeDimensionOverrides: {
        batch_size: 1, num_channels_latent: P.latentChannels, height_latent: P.latentSize, width_latent: P.latentSize,
      },
    });
  }
  /* ------------------------------- ops ------------------------------- */

  /**
   * Encode a tokenized prompt -> hidden states {data:Float32Array, dims}.
   * Feeds BOTH input_ids and attention_mask (WebGPU Attention kernel requires it).
   */
  async encodeText(session, { inputIds, attentionMask }) {
    const ort = await loadOrt();
    const dims = [1, inputIds.length];
    const feeds = {};
    const inNames = session.inputNames;
    const idName = inNames.find((n) => /input_ids|ids/i.test(n)) ?? inNames[0];
    const maskName = inNames.find((n) => /attention|mask/i.test(n));

    const build = (use64) => {
      feeds[idName] = new ort.Tensor(use64 ? 'int64' : 'int32',
        use64 ? BigInt64Array.from(inputIds, (v) => BigInt(v)) : Int32Array.from(inputIds), dims);
      if (maskName) {
        feeds[maskName] = new ort.Tensor(use64 ? 'int64' : 'int32',
          use64 ? BigInt64Array.from(attentionMask, (v) => BigInt(v)) : Int32Array.from(attentionMask), dims);
      }
    };

    let out;
    try {
      build(true);
      out = await session.run(feeds);
    } catch (err) {
      this._log.warn('text encoder int64 failed, retrying int32', { error: String(err?.message ?? err) });
      build(false);
      out = await session.run(feeds);
    }
    const key = session.outputNames.find((n) => /hidden/i.test(n)) ?? session.outputNames[0];
    const t = out[key];
    return { data: t.data, dims: t.dims };
  }

  /**
   * One UNet evaluation. Returns the epsilon (noise) prediction as Float32Array.
   * @param {Float32Array} sample  latent, [1,4,64,64] flattened
   * @param {number} timestep
   * @param {{data:Float32Array, dims:number[]}} hidden
   */
  async runUnet(session, { sample, timestep, hidden }) {
    const ort = await loadOrt();
    const inNames = session.inputNames;
    const sampleName = inNames.find((n) => /sample|latent/i.test(n)) ?? inNames[0];
    const tName = inNames.find((n) => /timestep|time/i.test(n));
    const hName = inNames.find((n) => /hidden|encoder/i.test(n));
    const feeds = {
      [sampleName]: new ort.Tensor('float32', sample, [1, P.latentChannels, P.latentSize, P.latentSize]),
    };
    if (hName) feeds[hName] = new ort.Tensor('float32', hidden.data, hidden.dims);
    if (tName) {
      // Try int64 scalar-ish [1]; fall back to float32 if the export wants that.
      try {
        feeds[tName] = new ort.Tensor('int64', BigInt64Array.from([BigInt(timestep)]), [1]);
        const out = await session.run(feeds);
        return pickOut(out, session);
      } catch (int64Err) {
        // Log the original error too — a float32 retry failing differently
        // means the real cause isn't the timestep dtype.
        this._log.warn('unet int64 timestep failed, retrying float32', {
          error: String(int64Err?.message ?? int64Err),
        });
        feeds[tName] = new ort.Tensor('float32', Float32Array.from([timestep]), [1]);
        try {
          const out = await session.run(feeds);
          return pickOut(out, session);
        } catch (float32Err) {
          throw new SMError(CODES.E_INTERNAL,
            'UNet rejected the timestep input as both int64 and float32.',
            {
              action: 'Reload and try again. If it persists, this model export may need a different timestep dtype than either fallback.',
              detail: {
                int64Error: String(int64Err?.message ?? int64Err),
                float32Error: String(float32Err?.message ?? float32Err),
              },
            });
        }
      }
    }
    const out = await session.run(feeds);
    return pickOut(out, session);
  }

  /** Decode a latent -> RGBA8 (width=height=imageSize). Applies 1/scaling_factor. */
  async decode(session, latent) {
    const ort = await loadOrt();
    const scaled = scaleLatentsForDecode(latent, P.vaeScaleFactor);
    const inName = session.inputNames.find((n) => /latent|sample/i.test(n)) ?? session.inputNames[0];
    const out = await session.run({ [inName]: new ort.Tensor('float32', scaled, [1, P.latentChannels, P.latentSize, P.latentSize]) });
    const key = session.outputNames.find((n) => /sample|image/i.test(n)) ?? session.outputNames[0];
    return nchwToRgba(out[key].data, P.imageSize, P.imageSize);
  }

}

function pickOut(out, session) {
  const key = session.outputNames.find((n) => /out_sample|sample|noise/i.test(n)) ?? session.outputNames[0];
  return out[key].data;
}
