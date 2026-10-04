import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  clampDuration, keyframeCountFor, totalRenderFrames, sdTurboFileList,
  totalDownloadBytes, LIMITS, MOTION, SDTURBO, rifeFileList, rifeDownloadBytes, RIFE,
} from '../src/engine/registry.js';

test('clampDuration enforces the 2..20s cap', () => {
  assert.equal(clampDuration(0), LIMITS.MIN_SECONDS);
  assert.equal(clampDuration(1), LIMITS.MIN_SECONDS);
  assert.equal(clampDuration(8), 8);
  assert.equal(clampDuration(20), 20);
  assert.equal(clampDuration(999), 20);
  assert.equal(clampDuration('abc'), LIMITS.DEFAULT_SECONDS);
});

test('keyframeCountFor stays within MIN..MAX and scales with length', () => {
  const k2 = keyframeCountFor(2);
  const k20 = keyframeCountFor(20);
  assert.ok(k2 >= MOTION.MIN_KEYFRAMES && k2 <= MOTION.MAX_KEYFRAMES);
  assert.ok(k20 >= MOTION.MIN_KEYFRAMES && k20 <= MOTION.MAX_KEYFRAMES);
  assert.ok(k20 >= k2);
});

test('totalRenderFrames = seconds * fps within cap', () => {
  assert.equal(totalRenderFrames(10), 10 * LIMITS.FPS);
  assert.equal(totalRenderFrames(100), 20 * LIMITS.FPS); // capped
});

test('sdTurboFileList has exactly the three required graphs', () => {
  const files = sdTurboFileList();
  assert.equal(files.length, 3);
  assert.ok(files.every((f) => f.required));
});

test('every file entry has an absolute URL and a key', () => {
  for (const f of sdTurboFileList()) {
    assert.ok(f.urls.length >= 1);
    assert.match(f.urls[0], /^https:\/\//);
    assert.ok(f.urls[0].endsWith('/model.onnx'));
    assert.match(f.key, /^sd-turbo\//);
    assert.ok(f.approxBytes > 0);
  }
});

test('pipeline constants are the SD2.1 (not SD1.x) values', () => {
  assert.equal(SDTURBO.pipeline.hiddenSize, 1024); // OpenCLIP-ViT-H, NOT 768
  assert.equal(SDTURBO.pipeline.vaeScaleFactor, 0.18215);
  assert.equal(SDTURBO.pipeline.latentChannels, 4);
  assert.equal(SDTURBO.pipeline.imageSize, 512);
  assert.equal(SDTURBO.pipeline.timestep, 999);
});

test('total download is the sum of all three required file sizes', () => {
  const sum = sdTurboFileList().reduce((a, f) => a + f.approxBytes, 0);
  assert.equal(totalDownloadBytes(), sum);
  assert.ok(totalDownloadBytes() > 0);
});

test('rifeFileList has exactly one required file with an absolute URL', () => {
  const files = rifeFileList();
  assert.equal(files.length, 1);
  assert.ok(files[0].required);
  assert.match(files[0].urls[0], /^https:\/\//);
  assert.match(files[0].key, /^rife\//);
});

test('rifeDownloadBytes is an order of magnitude smaller than SD-Turbo', () => {
  assert.ok(rifeDownloadBytes() > 0);
  assert.ok(rifeDownloadBytes() < totalDownloadBytes() / 10);
});

test('RIFE imageSize matches SDTURBO pipeline imageSize', () => {
  assert.equal(RIFE.imageSize, SDTURBO.pipeline.imageSize);
});
