import { test } from 'node:test';
import assert from 'node:assert/strict';
import { baseNoise, driftedNoise, anchorBlend, slerp, keyframePlan } from '../src/engine/motion.js';

test('baseNoise is deterministic for a seed and roughly standard-normal', () => {
  const a = baseNoise({ seed: 42, latentChannels: 4, latentSize: 64 });
  const b = baseNoise({ seed: 42, latentChannels: 4, latentSize: 64 });
  assert.deepEqual(a, b);
  assert.equal(a.length, 4 * 64 * 64);
  let mean = 0;
  for (const v of a) mean += v;
  mean /= a.length;
  assert.ok(Math.abs(mean) < 0.05, `mean ${mean}`);
});

test('driftedNoise at t=0 is (near) identity', () => {
  const base = baseNoise({ seed: 7, latentChannels: 4, latentSize: 8 });
  const d = driftedNoise(base, { latentChannels: 4, latentSize: 8, driftCells: 3, t: 0 });
  for (let i = 0; i < base.length; i += 1) assert.ok(Math.abs(d[i] - base[i]) < 1e-5);
});

test('driftedNoise at t=1 differs from base but preserves shape', () => {
  const base = baseNoise({ seed: 7, latentChannels: 4, latentSize: 16 });
  const d = driftedNoise(base, { latentChannels: 4, latentSize: 16, driftCells: 3, t: 1 });
  assert.equal(d.length, base.length);
  let diff = 0;
  for (let i = 0; i < base.length; i += 1) diff += Math.abs(d[i] - base[i]);
  assert.ok(diff > 0, 'drift should change the field');
});

test('anchorBlend at strength 0 = identity, at 1 = anchor', () => {
  const x = Float32Array.from([1, 2, 3]);
  const anchor = Float32Array.from([-1, -2, -3]);
  assert.deepEqual(Array.from(anchorBlend(x, anchor, 0)), [1, 2, 3]);
  assert.deepEqual(Array.from(anchorBlend(x, anchor, 1)), [-1, -2, -3]);
  const mid = anchorBlend(x, anchor, 0.5);
  assert.deepEqual(Array.from(mid), [0, 0, 0]);
});

test('slerp endpoints and midpoint stay on the arc', () => {
  const a = Float32Array.from([1, 0, 0, 0]);
  const b = Float32Array.from([0, 1, 0, 0]);
  const s0 = slerp(a, b, 0);
  const s1 = slerp(a, b, 1);
  for (let i = 0; i < a.length; i += 1) {
    assert.ok(Math.abs(s0[i] - a[i]) < 1e-5);
    assert.ok(Math.abs(s1[i] - b[i]) < 1e-5);
  }
  const mid = slerp(a, b, 0.5);
  const norm = Math.hypot(...mid);
  assert.ok(Math.abs(norm - 1) < 1e-4, `midpoint norm ${norm} should be ~1 (preserves magnitude)`);
});

test('slerp falls back to lerp for near-colinear vectors', () => {
  const a = Float32Array.from([1, 0.0001]);
  const b = Float32Array.from([1, 0.0002]);
  const mid = slerp(a, b, 0.5);
  assert.ok(Math.abs(mid[0] - 1) < 1e-3);
});

test('keyframePlan covers [0,1] and maps prompt segments', () => {
  const plan = keyframePlan(5, 3);
  assert.equal(plan.length, 5);
  assert.equal(plan[0].t, 0);
  assert.equal(plan[4].t, 1);
  // embed indices within range
  for (const p of plan) {
    assert.ok(p.embedA >= 0 && p.embedA <= 2);
    assert.ok(p.embedB >= 0 && p.embedB <= 2);
    assert.ok(p.embedT >= 0 && p.embedT <= 1.0001);
  }
});

test('keyframePlan single prompt keeps embed indices at 0', () => {
  const plan = keyframePlan(4, 1);
  for (const p of plan) { assert.equal(p.embedA, 0); assert.equal(p.embedB, 0); }
});
