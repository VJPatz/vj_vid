/**
 * postinstall: copy onnxruntime-web's prebuilt .wasm/.mjs into public/ort/,
 * so ORT runs offline instead of fetching from a CDN (see ort.js's wasmPaths).
 * No-op if onnxruntime-web isn't installed yet — the app still builds.
 */
import { cp, mkdir, readdir, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'node_modules', 'onnxruntime-web', 'dist');
const dest = join(root, 'public', 'ort');

async function main() {
  if (!existsSync(src)) {
    console.warn('[copy-ort-assets] onnxruntime-web/dist not found — skipping (ORT will use its CDN).');
    return;
  }
  await mkdir(dest, { recursive: true });
  const entries = await readdir(src);
  // Copy the whole dist rather than cherry-picking — ORT's file set changes
  // between minor versions; a missing .wasm is worse than a few extra KB.
  let copied = 0;
  for (const name of entries) {
    if (!/\.(wasm|mjs|js)$/.test(name)) continue;
    const from = join(src, name);
    if (!(await stat(from)).isFile()) continue;
    await cp(from, join(dest, name));
    copied += 1;
  }
  console.log(`[copy-ort-assets] copied ${copied} runtime files to public/ort/`);
}

main().catch((err) => {
  // Never fail the install over this — degrade to the CDN path.
  console.warn('[copy-ort-assets] non-fatal:', err?.message ?? err);
});
