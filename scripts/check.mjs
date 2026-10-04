/**
 * Zero-dependency static check: every src/test/scripts .mjs/.js parses cleanly,
 * and every RELATIVE import resolves to a real file with a matching named export.
 *
 * The cheap gate that runs anywhere — no browser, no GPU, no install of the heavy
 * deps. It catches the two mistakes that otherwise only surface at runtime in a
 * browser: a syntax error, and an import of a symbol that was renamed or removed.
 *
 * Syntax is checked with `node --check`, which understands ESM and never executes
 * the file (so importing onnxruntime-web etc. is fine — we never run it).
 */
import { readdir, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join, resolve, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const roots = ['src', 'test', 'scripts'];
const BARE = /^[^./]/; // bare specifier (npm package) — resolved by the bundler, not us

async function walk(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await walk(p)));
    else if (/\.(mjs|js)$/.test(entry.name)) out.push(p);
  }
  return out;
}

function exportsOf(text) {
  const names = new Set();
  const re = /export\s+(?:async\s+)?(?:function|class|const|let|var)\s+([A-Za-z0-9_$]+)/g;
  let m;
  while ((m = re.exec(text))) names.add(m[1]);
  for (const block of text.matchAll(/export\s*\{([^}]*)\}/g)) {
    for (const part of block[1].split(',')) {
      const as = part.split(/\sas\s/).pop().trim();
      if (as) names.add(as);
    }
  }
  if (/export\s+default/.test(text)) names.add('default');
  // re-exports: export { x } from './y' already captured above via the {} form
  return names;
}

function importsOf(text) {
  const out = [];
  const re = /import\s+(?:([A-Za-z0-9_$]+)\s*,?\s*)?(?:\{([^}]*)\})?\s*(?:\*\s*as\s*[A-Za-z0-9_$]+)?\s*from\s*['"]([^'"]+)['"]/g;
  let m;
  while ((m = re.exec(text))) {
    const named = (m[2] ?? '').split(',').map((s) => s.split(/\sas\s/)[0].trim()).filter(Boolean);
    out.push({ spec: m[3], named });
  }
  return out;
}

async function main() {
  const files = [];
  for (const r of roots) if (existsSync(join(root, r))) files.push(...(await walk(join(root, r))));

  let errors = 0;
  const cache = new Map();
  const readExports = async (abs) => {
    if (!cache.has(abs)) cache.set(abs, exportsOf(await readFile(abs, 'utf8')));
    return cache.get(abs);
  };

  for (const file of files) {
    const rel = file.slice(root.length + 1);

    // 1. syntax
    const res = spawnSync(process.execPath, ['--check', file], { encoding: 'utf8' });
    if (res.status !== 0) {
      console.error(`✗ ${rel}: syntax\n${(res.stderr || '').trim().split('\n').slice(0, 4).join('\n')}`);
      errors += 1;
      continue;
    }

    // 2. relative imports resolve + named exports exist
    const text = await readFile(file, 'utf8');
    for (const imp of importsOf(text)) {
      if (BARE.test(imp.spec)) continue;
      const target = resolve(dirname(file), imp.spec);
      const candidates = extname(target) ? [target] : [target + '.js', target + '.mjs', join(target, 'index.js')];
      const found = candidates.find((c) => existsSync(c));
      if (!found) {
        console.error(`✗ ${rel}: cannot resolve import '${imp.spec}'`);
        errors += 1;
        continue;
      }
      const exp = await readExports(found);
      for (const name of imp.named) {
        if (!exp.has(name)) {
          console.error(`✗ ${rel}: '${imp.spec}' has no export '${name}'`);
          errors += 1;
        }
      }
    }
  }

  if (errors) {
    console.error(`\ncheck failed: ${errors} problem(s) across ${files.length} files.`);
    process.exit(1);
  }
  console.log(`check ok: ${files.length} files parse, all relative imports resolve.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
