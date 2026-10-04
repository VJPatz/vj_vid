import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  scaledLinearAlphasCumprod, initNoiseSigma, epsilonToSample, addNoise, nchwToRgba,
} from '../src/engine/scheduler.js';

test('scaled_linear alphas_cumprod matches diffusers reference shape', () => {
  const ac = scaledLinearAlphasCumprod();
  assert.equal(ac.length, 1000);
  for (let i = 1; i < ac.length; i += 1) assert.ok(ac[i] < ac[i - 1] && ac[i] > 0 && ac[i] < 1);
  assert.ok(Math.abs(ac[0] - (1 - 0.00085)) < 1e-6);
  assert.ok(ac[999] > 0.003 && ac[999] < 0.006, `got ${ac[999]}`);
});

test('initNoiseSigma is exactly 1 for DDPM schedule', () => {
  assert.equal(initNoiseSigma(), 1.0);
});

test('epsilonToSample inverts addNoise (round trip)', () => {
  const ac = scaledLinearAlphasCumprod();
  const a = ac[500];
  const x0 = Float32Array.from([0.2, -0.5, 1.0, -1.0, 0.33]);
  const noise = Float32Array.from([0.1, 0.9, -0.4, 0.5, -0.2]);
  const xt = addNoise(x0, noise, a);
  const recovered = epsilonToSample(xt, noise, a);
  for (let i = 0; i < x0.length; i += 1) assert.ok(Math.abs(recovered[i] - x0[i]) < 1e-4, `idx ${i}: ${recovered[i]} vs ${x0[i]}`);
});

test('epsilonToSample rejects bad alpha', () => {
  assert.throws(() => epsilonToSample(new Float32Array(2), new Float32Array(2), 0));
  assert.throws(() => epsilonToSample(new Float32Array(2), new Float32Array(2), 1.5));
});

test('nchwToRgba de-interleaves channels correctly and sets opaque alpha', () => {
  const t = Float32Array.from([1.0, 0.0, -1.0]); // 1x1, R=1 G=0 B=-1
  const rgba = nchwToRgba(t, 1, 1);
  assert.equal(rgba[0], 255);
  assert.ok(Math.abs(rgba[1] - 128) <= 1);
  assert.equal(rgba[2], 0);
  assert.equal(rgba[3], 255);
});
