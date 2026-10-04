/**
 * Weight cache — resumable downloads + offline-after-first-run.
 * Backends: OPFS (resumable) > Cache API (resume-on-restart only) > memory (session-only).
 * A `.meta` sidecar tracks completion + ETag/length so a stale partial is discarded, not spliced.
 */

import { CODES, SMError, asSMError, throwIfAborted } from '../core/errors.js';
import { logger } from '../core/log.js';

const DIR = 'vj_vid-weights';
const CACHE = 'vj_vid-weights-v1';
const META = '.meta.json';
const CHUNK = 8 * 1024 * 1024;

export const BACKEND = Object.freeze({ OPFS: 'opfs', CACHE: 'cache', MEMORY: 'memory' });

export class WeightCache {
  constructor(backend, handles = {}) {
    this.backend = backend;
    this._dir = handles.dir ?? null;
    this._cache = handles.cache ?? null;
    this._mem = new Map();
    this._log = logger.child('cache');
  }

  get canResume() { return this.backend === BACKEND.OPFS; }
  get persistent() { return this.backend !== BACKEND.MEMORY; }

  static async open() {
    try {
      if (navigator.storage?.getDirectory) {
        const root = await navigator.storage.getDirectory();
        const dir = await root.getDirectoryHandle(DIR, { create: true });
        const probe = await dir.getFileHandle('.probe', { create: true });
        const w = await probe.createWritable();
        await w.write(new Uint8Array([1]));
        await w.close();
        await dir.removeEntry('.probe').catch(() => {});
        return new WeightCache(BACKEND.OPFS, { dir });
      }
    } catch (err) {
      logger.child('cache').warn('OPFS unavailable', { error: String(err?.message ?? err) });
    }
    try {
      if (typeof caches !== 'undefined') return new WeightCache(BACKEND.CACHE, { cache: await caches.open(CACHE) });
    } catch (err) {
      logger.child('cache').warn('Cache API unavailable', { error: String(err?.message ?? err) });
    }
    logger.child('cache').warn('no persistent storage — weights re-download each session');
    return new WeightCache(BACKEND.MEMORY);
  }

  async has(key) {
    return Boolean((await this._readMeta(key))?.complete);
  }

  /** True only if every given key is already fully cached. */
  async hasAll(keys) {
    for (const key of keys) {
      if (!(await this.has(key))) return false;
    }
    return true;
  }

  async get(key) {
    const meta = await this._readMeta(key);
    if (!meta?.complete) return null;
    const bytes = await this._readData(key);
    if (!bytes) return null;
    if (typeof meta.totalBytes === 'number' && meta.totalBytes > 0 && bytes.byteLength !== meta.totalBytes) {
      this._log.warn('cached size mismatch — discarding', { key });
      await this.delete(key);
      return null;
    }
    return bytes;
  }

  /** Fetch across candidate URLs (first to answer wins), using/updating the cache, resuming where supported. */
  async fetchWithCache({ urls, key, approxBytes = 0, onProgress = null, signal = null }) {
    throwIfAborted(signal);
    const cached = await this.get(key);
    if (cached) {
      onProgress?.({ key, phase: 'cached', loaded: cached.byteLength, total: cached.byteLength, fraction: 1 });
      return cached;
    }

    const { url, head } = await this._pickUrl(urls, signal);
    const total = head.contentLength ?? approxBytes ?? 0;

    let startAt = 0;
    if (this.canResume) {
      const meta = await this._readMeta(key);
      if (meta && !meta.complete) {
        const same = (!meta.etag || !head.etag || meta.etag === head.etag) &&
          (!meta.totalBytes || !head.contentLength || meta.totalBytes === head.contentLength);
        if (same && head.acceptsRanges) {
          startAt = await this._dataLength(key);
          if (startAt > 0) this._log.info('resuming', { key, from: startAt });
        } else {
          await this.delete(key);
        }
      }
    }

    const bytes = await this._download({ url, key, total, head, startAt, onProgress, signal });
    await this._writeMeta(key, { complete: true, totalBytes: bytes.byteLength, etag: head.etag ?? null, url, at: Date.now() });
    onProgress?.({ key, phase: 'complete', loaded: bytes.byteLength, total: bytes.byteLength, fraction: 1 });
    return bytes;
  }

  async delete(key) {
    try {
      if (this.backend === BACKEND.OPFS) {
        await this._dir.removeEntry(enc(key)).catch(() => {});
        await this._dir.removeEntry(enc(key) + META).catch(() => {});
      } else if (this.backend === BACKEND.CACHE) {
        await this._cache.delete(cacheUrl(key));
        await this._cache.delete(cacheUrl(key + META));
      } else {
        this._mem.delete(key);
        this._mem.delete(key + META);
      }
      return true;
    } catch {
      return false;
    }
  }

  async list() {
    const out = [];
    try {
      if (this.backend === BACKEND.OPFS) {
        for await (const [name, handle] of this._dir.entries()) {
          if (name.endsWith(META) || name.startsWith('.')) continue;
          const file = await handle.getFile();
          const meta = await this._readMeta(dec(name));
          out.push({ key: dec(name), bytes: file.size, complete: Boolean(meta?.complete) });
        }
      } else if (this.backend === BACKEND.CACHE) {
        for (const req of await this._cache.keys()) {
          const key = dec(decodeURIComponent(new URL(req.url).pathname.replace(/^\//, '')));
          if (key.endsWith(META)) continue;
          const res = await this._cache.match(req);
          out.push({ key, bytes: res ? (await res.arrayBuffer()).byteLength : 0, complete: true });
        }
      } else {
        for (const [key, val] of this._mem) {
          if (key.endsWith(META)) continue;
          out.push({ key, bytes: val.byteLength ?? 0, complete: true });
        }
      }
    } catch (err) {
      this._log.warn('list failed', { error: String(err?.message ?? err) });
    }
    return out.sort((a, b) => (a.key < b.key ? -1 : 1));
  }

  async totalBytes() {
    return (await this.list()).reduce((s, e) => s + e.bytes, 0);
  }

  async clear() {
    let n = 0;
    for (const e of await this.list()) if (await this.delete(e.key)) n += 1;
    return n;
  }

  async quota() {
    try {
      const est = await navigator.storage?.estimate?.();
      return est ? { usage: est.usage ?? null, quota: est.quota ?? null } : null;
    } catch {
      return null;
    }
  }

  async requestPersistence() {
    try {
      if (await navigator.storage?.persisted?.()) return true;
      return Boolean(await navigator.storage?.persist?.());
    } catch {
      return false;
    }
  }

  /* ---------------------------------------------------------------- */

  async _pickUrl(urls, signal) {
    let lastHead = null;
    for (const url of urls) {
      const head = await this._probe(url, signal);
      if (head.ok || head.status === 0) return { url, head }; // 0 = HEAD blocked, GET may still work
      lastHead = head;
    }
    throw new SMError(CODES.E_HTTP_STATUS, `No candidate URL for this file responded (last status ${lastHead?.status}).`, {
      action: 'The model repository layout may have changed. Set a working base URL in Settings → Model source.',
      detail: { urls },
    });
  }

  async _probe(url, signal) {
    try {
      const res = await fetch(url, { method: 'HEAD', signal: signal ?? undefined, redirect: 'follow' });
      if (res.ok) {
        const len = Number(res.headers.get('content-length'));
        return {
          ok: true, status: res.status,
          contentLength: Number.isFinite(len) && len > 0 ? len : null,
          etag: res.headers.get('etag'),
          acceptsRanges: (res.headers.get('accept-ranges') ?? '').toLowerCase().includes('bytes'),
        };
      }
      return { ok: false, status: res.status, contentLength: null, etag: null, acceptsRanges: false };
    } catch {
      return { ok: false, status: 0, contentLength: null, etag: null, acceptsRanges: false };
    }
  }

  async _download({ url, key, total, head, startAt, onProgress, signal }) {
    const started = Date.now();
    const useRange = this.canResume && head.acceptsRanges && total > CHUNK;
    if (!useRange) {
      const bytes = await this._stream({ url, key, total, onProgress, signal, started });
      await this._writeData(key, bytes);
      return bytes;
    }
    // Chunked Range into OPFS — peak memory is one chunk, not the whole file.
    let offset = startAt;
    const handle = await this._dir.getFileHandle(enc(key), { create: true });
    while (offset < total) {
      throwIfAborted(signal);
      const end = Math.min(offset + CHUNK, total) - 1;
      let res;
      try {
        res = await fetch(url, { headers: { Range: `bytes=${offset}-${end}` }, signal: signal ?? undefined, redirect: 'follow' });
      } catch (err) {
        throw asSMError(err, CODES.E_NETWORK_FAILED, `Download of "${key}" failed.`, {
          action: 'Check your connection and press Generate again — it resumes where it stopped.',
        });
      }
      if (res.status !== 206 && res.status !== 200) {
        throw new SMError(CODES.E_HTTP_STATUS, `Unexpected status ${res.status} downloading "${key}".`, {
          action: 'Try again shortly. The host may be rate-limiting, or the file moved.',
        });
      }
      const chunk = new Uint8Array(await res.arrayBuffer());
      const w = await handle.createWritable({ keepExistingData: true });
      await w.write({ type: 'write', position: offset, data: chunk });
      await w.close();
      offset += chunk.byteLength;
      await this._writeMeta(key, { complete: false, totalBytes: total, downloaded: offset, etag: head.etag ?? null, url, at: Date.now() });
      emit(onProgress, { key, phase: 'downloading', loaded: offset, total, started });
      if (res.status === 200 && chunk.byteLength >= total) break; // server ignored Range header
      if (chunk.byteLength === 0) throw new SMError(CODES.E_NETWORK_FAILED, `Download of "${key}" stalled.`, { action: 'Try again — it resumes.' });
    }
    const bytes = await this._readData(key);
    if (!bytes) throw new SMError(CODES.E_STORAGE_UNAVAILABLE, `Could not read back "${key}".`, { action: 'Free disk space and try again.' });
    return bytes;
  }

  async _stream({ url, key, total, onProgress, signal, started }) {
    let res;
    try {
      res = await fetch(url, { signal: signal ?? undefined, redirect: 'follow' });
    } catch (err) {
      throw asSMError(err, CODES.E_NETWORK_FAILED, `Download of "${key}" failed.`, { action: 'Check your connection and try again.' });
    }
    if (!res.ok) {
      throw new SMError(CODES.E_HTTP_STATUS, `Unexpected status ${res.status} downloading "${key}".`, {
        action: res.status === 404 ? 'Set a working base URL in Settings → Model source.' : 'Try again shortly.',
      });
    }
    const declared = Number(res.headers.get('content-length'));
    const expected = Number.isFinite(declared) && declared > 0 ? declared : total;
    if (!res.body) {
      const buf = new Uint8Array(await res.arrayBuffer());
      emit(onProgress, { key, phase: 'downloading', loaded: buf.byteLength, total: buf.byteLength, started });
      return buf;
    }
    const reader = res.body.getReader();
    const chunks = [];
    let loaded = 0;
    try {
      for (;;) {
        throwIfAborted(signal);
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        emit(onProgress, { key, phase: 'downloading', loaded, total: expected, started });
      }
    } catch (err) {
      await reader.cancel().catch(() => {});
      throw asSMError(err, CODES.E_NETWORK_FAILED, `Download of "${key}" was interrupted.`, { action: 'Check your connection and try again.' });
    }
    const bytes = concat(chunks, loaded);
    if (expected > 0 && bytes.byteLength < expected * 0.999) {
      throw new SMError(CODES.E_SIZE_MISMATCH, `Download of "${key}" was truncated.`, { action: 'Check your connection and try again.' });
    }
    if (looksLikeHtml(bytes)) {
      throw new SMError(CODES.E_HTTP_STATUS, `The server returned a web page instead of model data for "${key}".`, {
        action: 'The file path is wrong. Check Settings → Model source.',
      });
    }
    return bytes;
  }

  async _readData(key) {
    if (this.backend === BACKEND.OPFS) {
      try {
        const h = await this._dir.getFileHandle(enc(key));
        return new Uint8Array(await (await h.getFile()).arrayBuffer());
      } catch {
        return null;
      }
    }
    if (this.backend === BACKEND.CACHE) {
      const res = await this._cache.match(cacheUrl(key));
      return res ? new Uint8Array(await res.arrayBuffer()) : null;
    }
    return this._mem.get(key) ?? null;
  }

  async _dataLength(key) {
    if (this.backend === BACKEND.OPFS) {
      try {
        return (await (await this._dir.getFileHandle(enc(key))).getFile()).size;
      } catch {
        return 0;
      }
    }
    return (await this._readData(key))?.byteLength ?? 0;
  }

  async _writeData(key, bytes) {
    if (this.backend === BACKEND.OPFS) {
      const h = await this._dir.getFileHandle(enc(key), { create: true });
      const w = await h.createWritable();
      await w.write(bytes);
      await w.close();
      return;
    }
    if (this.backend === BACKEND.CACHE) {
      await this._cache.put(cacheUrl(key), new Response(bytes, { headers: { 'content-type': 'application/octet-stream', 'content-length': String(bytes.byteLength) } }));
      return;
    }
    this._mem.set(key, bytes);
  }

  async _readMeta(key) {
    try {
      if (this.backend === BACKEND.OPFS) {
        const h = await this._dir.getFileHandle(enc(key) + META);
        return JSON.parse(await (await h.getFile()).text());
      }
      if (this.backend === BACKEND.CACHE) {
        const res = await this._cache.match(cacheUrl(key + META));
        return res ? await res.json() : null;
      }
      return this._mem.get(key + META) ?? null;
    } catch {
      return null;
    }
  }

  async _writeMeta(key, meta) {
    try {
      if (this.backend === BACKEND.OPFS) {
        const h = await this._dir.getFileHandle(enc(key) + META, { create: true });
        const w = await h.createWritable();
        await w.write(new TextEncoder().encode(JSON.stringify(meta)));
        await w.close();
        return;
      }
      if (this.backend === BACKEND.CACHE) {
        await this._cache.put(cacheUrl(key + META), new Response(JSON.stringify(meta), { headers: { 'content-type': 'application/json' } }));
        return;
      }
      this._mem.set(key + META, meta);
    } catch (err) {
      this._log.warn('could not persist meta', { key, error: String(err?.message ?? err) });
    }
  }
}

function emit(onProgress, { key, phase, loaded, total, started }) {
  if (!onProgress) return;
  const elapsed = Math.max(1, Date.now() - started) / 1000;
  const bps = loaded / elapsed;
  const remaining = Math.max(0, (total || 0) - loaded);
  onProgress({
    key, phase, loaded, total: total || null,
    fraction: total > 0 ? Math.min(1, loaded / total) : null,
    bytesPerSecond: bps,
    etaSeconds: bps > 0 && total > 0 ? remaining / bps : null,
  });
}

function concat(chunks, total) {
  const out = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.byteLength; }
  return out;
}

function looksLikeHtml(bytes) {
  if (bytes.byteLength < 14) return false;
  const head = new TextDecoder('utf-8', { fatal: false }).decode(bytes.subarray(0, 64)).trim().toLowerCase();
  return head.startsWith('<!doctype html') || head.startsWith('<html');
}

const enc = (k) => String(k).replace(/[/\\]/g, '__');
const dec = (n) => String(n).replace(/__/g, '/');
const cacheUrl = (k) => `https://vj-vid.local/${encodeURIComponent(k)}`;
