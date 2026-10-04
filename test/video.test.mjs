import { test } from 'node:test';
import assert from 'node:assert/strict';
import { interpolateBetween } from '../src/engine/video.js';
import { buildPromptSchedule } from '../src/engine/prompt.js';
import { mulberry32, hashSeed, gaussianNoise } from '../src/core/rng.js';

// Fake RIFE: averages the two frames' first byte as a marker — tests call
// count/ordering/shape without a real ONNX session.
function makeFakeRife() {
  let calls = 0;
  const rife = {
    calls: () => calls,
    async interpolate(_session, a, b, t) {
      calls += 1;
      const out = new Uint8ClampedArray(4);
      out[0] = Math.round(a[0] + (b[0] - a[0]) * t);
      out[3] = 255;
      return out;
    },
  };
  return rife;
}

test('interpolateBetween: returns exactly `count` frames in order', async () => {
  const rife = makeFakeRife();
  const a = new Uint8ClampedArray([0, 0, 0, 255]);
  const b = new Uint8ClampedArray([100, 0, 0, 255]);
  const frames = await interpolateBetween(rife, {}, a, b, 7, 1, 1);
  assert.equal(frames.length, 7);
  // monotonic toward b even though RIFE calls happen midpoint-first
  for (let i = 1; i < frames.length; i += 1) {
    assert.ok(frames[i][0] >= frames[i - 1][0], `frame ${i} should be >= frame ${i - 1}`);
  }
});

test('interpolateBetween: count=0 returns empty array without calling RIFE', async () => {
  const rife = makeFakeRife();
  const a = new Uint8ClampedArray([0, 0, 0, 255]);
  const b = new Uint8ClampedArray([100, 0, 0, 255]);
  const frames = await interpolateBetween(rife, {}, a, b, 0, 1, 1);
  assert.equal(frames.length, 0);
  assert.equal(rife.calls(), 0);
});

test('interpolateBetween: count=1 returns the exact midpoint with one RIFE call', async () => {
  const rife = makeFakeRife();
  const a = new Uint8ClampedArray([0, 0, 0, 255]);
  const b = new Uint8ClampedArray([100, 0, 0, 255]);
  const frames = await interpolateBetween(rife, {}, a, b, 1, 1, 1);
  assert.equal(frames.length, 1);
  assert.equal(rife.calls(), 1);
  assert.equal(frames[0][0], 50);
});

test('interpolateBetween: reuses shared midpoints instead of recomputing them', async () => {
  const rife = makeFakeRife();
  const a = new Uint8ClampedArray([0, 0, 0, 255]);
  const b = new Uint8ClampedArray([100, 0, 0, 255]);
  // count=3 needs exactly 3 calls (mid, then each half's mid) — no redundant recompute
  await interpolateBetween(rife, {}, a, b, 3, 1, 1);
  assert.equal(rife.calls(), 3);
});

test('buildPromptSchedule keeps the user scene verbatim in every beat', () => {
  const scene = 'a neon city street in the rain';
  const beats = buildPromptSchedule(scene);
  assert.ok(beats.length >= 1);
  for (const b of beats) assert.ok(b.includes(scene), `beat missing scene: ${b}`);
});

test('buildPromptSchedule handles empty prompt without throwing', () => {
  const beats = buildPromptSchedule('   ');
  assert.ok(beats.length >= 1 && beats[0].length > 0);
});

test('rng is deterministic and seed-stable', () => {
  const r1 = mulberry32(123);
  const r2 = mulberry32(123);
  for (let i = 0; i < 100; i += 1) assert.equal(r1(), r2());
  assert.equal(hashSeed('sunrise'), hashSeed('sunrise'));
  assert.notEqual(hashSeed('sunrise'), hashSeed('sunset'));
  const g1 = gaussianNoise(1000, 5);
  const g2 = gaussianNoise(1000, 5);
  assert.deepEqual(g1, g2);
});
