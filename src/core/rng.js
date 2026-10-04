/** Seeded, deterministic RNG — same (prompt, seed) always reproduces the same video. Not for crypto. */

/** A 32-bit seeded uniform generator in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Turn an arbitrary string into a 32-bit seed (xfnv-1a). */
export function hashSeed(str) {
  let h = 2166136261 >>> 0;
  const s = String(str);
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** Fill a Float32Array with standard-normal samples via Box-Muller. */
export function gaussianNoise(n, seed) {
  const rng = mulberry32(seed >>> 0);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(rng(), 1e-7);
    const v = rng();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}
