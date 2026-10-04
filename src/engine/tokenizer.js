/**
 * CLIP text tokenizer, via @huggingface/transformers.
 * Loads openai/clip-vit-large-patch14, not sd-turbo's own tokenizer/ (legacy
 * vocab.json+merges.txt format, unloadable by transformers.js). Safe
 * substitute: OpenCLIP-ViT-H's BPE vocab/merges are byte-identical to
 * OpenAI CLIP's — only the model weights differ, not the tokenizer.
 * Needs both input_ids AND attention_mask — WebGPU's Attention kernel
 * throws on a padded sequence without the mask.
 */

import { CODES, asSMError } from '../core/errors.js';
import { logger } from '../core/log.js';
import { SDTURBO } from './registry.js';

let tokenizerPromise = null;

export function loadTokenizer(repo = SDTURBO.tokenizerRepo) {
  if (!tokenizerPromise) {
    tokenizerPromise = (async () => {
      let T;
      try {
        T = await import('@huggingface/transformers');
      } catch (err) {
        throw asSMError(err, CODES.E_TOKENIZER_FAILED, 'Could not load the tokenizer library.', {
          action: 'Run `npm install` and reload.',
        });
      }
      // Only the tokenizer.json is fetched (a few hundred KB), not model weights.
      try {
        T.env.allowLocalModels = false;
        const tok = await T.AutoTokenizer.from_pretrained(repo);
        logger.child('tokenizer').info('ready', { repo });
        return tok;
      } catch (err) {
        throw asSMError(err, CODES.E_TOKENIZER_FAILED, 'Could not load the CLIP tokenizer.', {
          action: 'Check your connection on first run — the tokenizer is a small one-time download.',
          detail: { repo },
        });
      }
    })();
  }
  return tokenizerPromise;
}

/** Tokenize to fixed length with padding + truncation. */
export async function tokenize(text, { maxTokens = SDTURBO.pipeline.maxTokens } = {}) {
  const tok = await loadTokenizer();
  const out = tok(text ?? '', { padding: 'max_length', max_length: maxTokens, truncation: true, return_tensor: false });
  const ids = normalize(out.input_ids, maxTokens);
  const mask = normalize(out.attention_mask ?? ids.map((v) => (v === 0 ? 0 : 1)), maxTokens);
  return { inputIds: ids, attentionMask: mask };
}

function normalize(arr, maxTokens) {
  // transformers.js can return a nested [ [...] ] for a single string; flatten.
  let a = Array.isArray(arr?.[0]) ? arr[0] : arr;
  a = Array.from(a, (v) => Number(v));
  if (a.length > maxTokens) a = a.slice(0, maxTokens);
  while (a.length < maxTokens) a.push(0);
  return a;
}
