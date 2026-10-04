/** UI controller — owns the DOM, wires pipeline events to it. No heavy logic here; see src/engine. Framework-free. */

import { el, formatBytes, formatDuration } from './dom.js';
import { probe, TIER } from '../engine/capability.js';
import { Pipeline, PHASE } from '../engine/pipeline.js';
import { WeightCache } from '../engine/cache.js';
import {
  LIMITS, MOTION, clampDuration, keyframeCountFor, totalDownloadBytes, rifeDownloadBytes,
  SDTURBO, RIFE,
} from '../engine/registry.js';
import { logger } from '../core/log.js';
import { SEVERITY } from '../core/errors.js';

// Motion tuning overrides via URL — for calibrating visually without a
// rebuild. Flat: ?anchor=0.35&drift=6&mode=chain. Ramped across the clip:
// ?anchorStart=0.5&anchorEnd=0.2&driftStart=2&driftEnd=10 (start/end win over
// the flat form if both given). Falls back to registry.js's MOTION when
// absent or out of range. Not persisted — query params only.
function motionFromQuery() {
  const q = new URLSearchParams(location.search);
  const num = (key) => (q.has(key) ? Number(q.get(key)) : NaN); // absent -> NaN, not 0
  const anchor = num('anchor');
  const drift = num('drift');
  const anchorStart = num('anchorStart');
  const anchorEnd = num('anchorEnd');
  const driftStart = num('driftStart');
  const driftEnd = num('driftEnd');
  const mode = q.get('mode');
  const isUnit = (n) => Number.isFinite(n) && n >= 0 && n <= 1;
  const isPos = (n) => Number.isFinite(n) && n >= 0;

  const baseAnchor = isUnit(anchor) ? anchor : MOTION.ANCHOR_STRENGTH;
  const baseDrift = isPos(drift) ? drift : MOTION.LATENT_DRIFT;
  const motion = {
    ANCHOR_MODE: mode === 'first' || mode === 'chain' ? mode : MOTION.ANCHOR_MODE,
    ANCHOR_STRENGTH: baseAnchor,
    LATENT_DRIFT: baseDrift,
    ANCHOR_STRENGTH_START: isUnit(anchorStart) ? anchorStart : baseAnchor,
    ANCHOR_STRENGTH_END: isUnit(anchorEnd) ? anchorEnd : baseAnchor,
    LATENT_DRIFT_START: isPos(driftStart) ? driftStart : baseDrift,
    LATENT_DRIFT_END: isPos(driftEnd) ? driftEnd : baseDrift,
  };
  const overridden = motion.ANCHOR_MODE !== MOTION.ANCHOR_MODE
    || motion.ANCHOR_STRENGTH_START !== MOTION.ANCHOR_STRENGTH_START
    || motion.ANCHOR_STRENGTH_END !== MOTION.ANCHOR_STRENGTH_END
    || motion.LATENT_DRIFT_START !== MOTION.LATENT_DRIFT_START
    || motion.LATENT_DRIFT_END !== MOTION.LATENT_DRIFT_END;
  return { motion, overridden };
}

const PHASE_LABEL_COLD = {
  [PHASE.PROBE]: 'Checking your hardware…',
  [PHASE.DOWNLOAD]: 'Downloading model (first run only)…',
  [PHASE.ENCODE_TEXT]: 'Reading your prompt…',
  [PHASE.KEYFRAMES]: 'Generating keyframes…',
  [PHASE.DECODE]: 'Developing images…',
  [PHASE.RIFE_DOWNLOAD]: 'Downloading motion model (first run only)…',
  [PHASE.ENCODE_VIDEO]: 'Interpolating motion and encoding video…',
  [PHASE.DONE]: 'Done',
};

// Warm run: DOWNLOAD/RIFE_DOWNLOAD are instant cache checks, not transfers.
const PHASE_LABEL_WARM = {
  ...PHASE_LABEL_COLD,
  [PHASE.DOWNLOAD]: 'Loading cached model…',
  [PHASE.RIFE_DOWNLOAD]: 'Loading cached motion model…',
};

const MODE = Object.freeze({ IMAGE: 'image', VIDEO: 'video' });
const MODE_LABEL = { [MODE.IMAGE]: 'Image', [MODE.VIDEO]: 'Video' };

const THEME_KEY = 'vj_vid.theme.v1';

export class App {
  constructor(root) {
    this.root = root;
    this.pipeline = new Pipeline();
    this.running = false;
    this.lastResult = null;
    this.mode = MODE.IMAGE;
    this._build();
    this._wirePipeline();
    this._initTheme();
    this._init();
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !this.settings.panel.hidden) this._toggleSettings(false);
    });
  }

  /* ------------------------------- layout ------------------------------- */

  _build() {
    this.root.innerHTML = '';
    this.root.dataset.boot = 'ready';

    // header
    const header = el('header', { class: 'sm-header' }, [
      el('div', { class: 'sm-brand' }, [
        el('span', { class: 'sm-logo', text: 'v_v' }),
        el('div', {}, [
          el('h1', { text: 'vj_vid' }),
          el('p', { class: 'sm-tag', text: 'Image and video generation, entirely in your browser. No servers.' }),
        ]),
      ]),
      el('div', { class: 'sm-header-actions' }, [
        this.themeBtn = el('button', {
          class: 'sm-icon-btn', 'aria-label': 'Toggle theme', onClick: () => this._toggleTheme(),
        }, ['◐']),
        el('button', { class: 'sm-ghost', onClick: () => this._toggleSettings() }, ['Settings']),
      ]),
    ]);

    // prompt
    this.promptInput = el('textarea', {
      class: 'sm-prompt', rows: 3, placeholder: 'a misty mountain landscape at dawn, sun breaking through fog…',
      oninput: () => this._updateEstimate(),
    });

    // mode select
    this.modeSelect = el('select', {
      class: 'sm-seed', onChange: (e) => this._setMode(e.target.value),
    }, Object.values(MODE).map((m) => el('option', { value: m, text: MODE_LABEL[m] })));
    this.modeSelect.value = this.mode;

    // controls (length/seed) — length only applies to Video.
    this.secondsInput = el('input', {
      type: 'range', min: String(LIMITS.MIN_SECONDS), max: String(LIMITS.MAX_SECONDS),
      value: String(LIMITS.DEFAULT_SECONDS), step: '1', class: 'sm-range', oninput: () => this._updateEstimate(),
    });
    this.secondsOut = el('output', { class: 'sm-secout' }, [`${LIMITS.DEFAULT_SECONDS}s`]);
    this.seedInput = el('input', { type: 'text', class: 'sm-seed', placeholder: 'seed (optional)' });

    this.generateBtn = el('button', { class: 'sm-primary', onClick: () => this._onGenerate() }, ['Generate']);
    this.cancelBtn = el('button', { class: 'sm-ghost', hidden: 'hidden', onClick: () => this._onCancel() }, ['Cancel']);
    this.estimate = el('p', { class: 'sm-estimate' });

    this.lengthField = el('label', { class: 'sm-field' }, [
      el('span', { class: 'sm-label' }, ['Length', this.secondsOut]),
      this.secondsInput,
    ]);

    const controls = el('div', { class: 'sm-controls' }, [
      el('label', { class: 'sm-field' }, [el('span', { class: 'sm-label', text: 'Mode' }), this.modeSelect]),
      el('label', { class: 'sm-field' }, [el('span', { class: 'sm-label', text: 'Seed' }), this.seedInput]),
      this.lengthField,
    ]);

    // progress
    this.progressBar = el('div', { class: 'sm-bar-fill' });
    this.progressWrap = el('div', { class: 'sm-bar', hidden: 'hidden' }, [this.progressBar]);
    this.statusLine = el('p', { class: 'sm-status' });

    // preview
    this.video = el('video', { class: 'sm-video', controls: 'controls', playsinline: 'playsinline', loop: 'loop', hidden: 'hidden' });
    this.poster = el('div', { class: 'sm-poster' }, [el('span', { text: 'Your output will appear here' })]);
    this.downloadBtn = el('a', { class: 'sm-primary sm-download', hidden: 'hidden', download: 'vj_vid.mp4' }, ['Download .mp4']);
    this.keyframeStrip = el('div', { class: 'sm-strip', hidden: 'hidden' });

    // settings panel
    this.settings = this._buildSettings();

    // fatal banner
    this.banner = el('div', { class: 'sm-banner', hidden: 'hidden' });

    this.root.append(
      header,
      this.banner,
      el('main', { class: 'sm-main' }, [
        el('section', { class: 'sm-panel sm-compose' }, [
          el('label', { class: 'sm-label', text: 'Prompt' }),
          this.promptInput,
          controls,
          this.estimate,
          el('div', { class: 'sm-actions' }, [this.generateBtn, this.cancelBtn]),
          this.progressWrap,
          this.statusLine,
        ]),
        el('section', { class: 'sm-panel sm-preview' }, [
          el('div', { class: 'sm-preview-frame' }, [this.poster, this.video]),
          this.keyframeStrip,
          el('div', { class: 'sm-actions' }, [this.downloadBtn]),
        ]),
      ]),
      this.settings.backdrop,
      this.settings.panel,
      el('footer', { class: 'sm-footer' }, [
        el('span', {}, [
          `Models: ${SDTURBO.label} · `, el('a', { href: SDTURBO.licenseUrl, target: '_blank', rel: 'noreferrer', text: SDTURBO.license }), ' (non-commercial) · ',
          `${RIFE.label} · `, el('a', { href: RIFE.licenseUrl, target: '_blank', rel: 'noreferrer', text: RIFE.license }),
        ]),
      ]),
    );

    this._setMode(this.mode);
    this._updateEstimate();
  }

  _buildSettings() {
    const cacheInfo = el('p', { class: 'sm-muted', text: 'Checking cache…' });
    const backdrop = el('div', { class: 'sm-settings-backdrop', hidden: 'hidden', onClick: () => this._toggleSettings(false) });
    const panel = el('aside', { class: 'sm-settings', hidden: 'hidden' }, [
      el('div', { class: 'sm-settings-head' }, [
        el('h2', { text: 'Settings' }),
        el('button', { class: 'sm-icon-btn', 'aria-label': 'Close settings', onClick: () => this._toggleSettings(false) }, ['✕']),
      ]),
      el('h3', { text: 'Cached weights' }),
      cacheInfo,
      el('button', { class: 'sm-ghost', onClick: () => this._clearCache() }, ['Clear cache']),
    ]);
    return { panel, backdrop, cacheInfo };
  }

  /* ------------------------------- theme ------------------------------- */

  _initTheme() {
    let saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch { /* private mode etc. */ }
    if (saved === 'light' || saved === 'dark') this._applyTheme(saved);
    else this._applyTheme(null); // follow system
  }

  _applyTheme(theme) {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
    this.themeBtn.textContent = theme === 'dark' ? '☾' : theme === 'light' ? '☀' : '◐';
  }

  _toggleTheme() {
    const current = document.documentElement.dataset.theme
      || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = current === 'dark' ? 'light' : 'dark';
    this._applyTheme(next);
    try { localStorage.setItem(THEME_KEY, next); } catch { /* ignore */ }
  }

  /* ------------------------------- logic ------------------------------- */

  async _init() {
    const result = await probe();
    logger.child('ui').info('probe', { tier: result.tier, ms: result.durationMs });
    if (result.fatal) {
      const fatal = result.reasons.find((r) => r.severity === SEVERITY.FATAL);
      this._showBanner(`${fatal.message} ${fatal.action}`, true);
      this.generateBtn.disabled = true;
    } else {
      const degrade = result.reasons.find((r) => r.severity === SEVERITY.DEGRADE);
      if (degrade) this._showBanner(`${degrade.message} ${degrade.action}`, false);
    }
    this._refreshCacheInfo();
  }

  async _refreshCacheInfo() {
    try {
      const cache = await WeightCache.open();
      // Only this app's tracked files — storage.estimate()'s quota covers the whole origin.
      const bytes = await cache.totalBytes();
      this.settings.cacheInfo.textContent = bytes > 0
        ? `${formatBytes(bytes)} cached · ${cache.persistent ? 'persistent' : 'session-only'}`
        : `Nothing cached yet. First run downloads ~${formatBytes(totalDownloadBytes() + rifeDownloadBytes())}.`;
    } catch {
      this.settings.cacheInfo.textContent = 'Cache status unavailable.';
    }
  }

  async _clearCache() {
    const cache = await WeightCache.open();
    const n = await cache.clear();
    this._flash(`Cleared ${n} file(s).`);
    this._refreshCacheInfo();
  }

  _setMode(mode) {
    this.mode = mode;
    this.modeSelect.value = mode;
    this.lengthField.hidden = mode === MODE.IMAGE;
    this._updateEstimate();
  }

  _updateEstimate() {
    if (this.mode === MODE.IMAGE) {
      this.estimate.textContent = `1 image · ${LIMITS.WIDTH}×${LIMITS.HEIGHT}`;
      return;
    }
    const secs = clampDuration(this.secondsInput.value);
    this.secondsOut.textContent = `${secs}s`;
    const k = keyframeCountFor(secs);
    this.estimate.textContent = `${k} keyframes · ${secs * LIMITS.FPS} frames @ ${LIMITS.FPS}fps · ${LIMITS.WIDTH}×${LIMITS.HEIGHT}`;
  }

  _wirePipeline() {
    let warmRun = false;
    let phaseLabel = '';
    let downloadBytes = null; // { loaded, total } while phase === 'download'/'rife_download'
    const render = (overallFraction) => {
      const pct = Math.round(Math.min(1, Math.max(0, overallFraction)) * 100);
      this.progressBar.style.width = `${pct}%`;
      const bytesPart = downloadBytes && downloadBytes.total
        ? ` (${formatBytes(downloadBytes.loaded)} / ${formatBytes(downloadBytes.total)})`
        : '';
      this.statusLine.textContent = `${phaseLabel} ${pct}%${bytesPart}`;
    };
    this.pipeline.on('phase', ({ phase, warm }) => {
      if (typeof warm === 'boolean') warmRun = warm;
      const labels = warmRun ? PHASE_LABEL_WARM : PHASE_LABEL_COLD;
      if (labels[phase]) phaseLabel = labels[phase];
      if (phase !== 'download' && phase !== 'rife_download') downloadBytes = null;
    });
    this.pipeline.on('progress', ({ overall }) => {
      render(overall);
    });
    this.pipeline.on('download', ({ warm, loadedAll, totalAll }) => {
      phaseLabel = warm ? 'Loading cached model…' : 'Downloading model (first run only)…';
      downloadBytes = totalAll ? { loaded: loadedAll, total: totalAll } : null;
    });
    this.pipeline.on('keyframe', ({ index, count }) => {
      phaseLabel = `Generating keyframes… (${index + 1}/${count})`;
    });
    this.pipeline.on('error', ({ error }) => {
      this._showBanner(`${error.message}${error.action ? ' — ' + error.action : ''}`, false);
    });
  }

  _resetOutputs() {
    this.banner.hidden = true;
    this.video.pause?.();
    this.video.removeAttribute('src');
    this.video.hidden = true;
    this.downloadBtn.hidden = true;
    this.downloadBtn.download = 'vj_vid.mp4';
    this.downloadBtn.textContent = 'Download .mp4';
    this.keyframeStrip.hidden = true;
    this.keyframeStrip.innerHTML = '';
    this.poster.hidden = true;
  }

  async _onGenerate() {
    if (this.running) return;
    const prompt = this.promptInput.value.trim();
    if (!prompt) { this._flash('Enter a prompt first.'); this.promptInput.focus(); return; }

    this.running = true;
    this.generateBtn.disabled = true;
    this.cancelBtn.hidden = false;
    this.progressWrap.hidden = false;
    this.progressBar.style.width = '0%';
    this._resetOutputs();

    try {
      if (this.mode === MODE.IMAGE) await this._runImage(prompt);
      else await this._runVideo(prompt);
    } catch (err) {
      if (err?.code === 'E_CANCELLED') this.statusLine.textContent = 'Cancelled.';
      // other errors already surfaced via the pipeline's 'error' event
    } finally {
      this.running = false;
      this.generateBtn.disabled = false;
      this.cancelBtn.hidden = true;
    }
  }

  _onCancel() {
    this.pipeline.cancel();
  }

  async _runImage(prompt) {
    const { image, meta } = await this.pipeline.generateImage({
      prompt,
      seed: this.seedInput.value.trim() || null,
    });
    const c = el('canvas', { width: String(image.width), height: String(image.height), class: 'sm-thumb-full' });
    c.getContext('2d').putImageData(image, 0, 0);
    this.poster.innerHTML = '';
    this.poster.append(c);
    this.poster.hidden = false;
    c.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      this.downloadBtn.href = url;
      this.downloadBtn.download = 'vj_vid.png';
      this.downloadBtn.textContent = 'Download .png';
      this.downloadBtn.hidden = false;
    });
    this.statusLine.textContent = `Image done in ${formatDuration(meta.ms)}. Seed ${meta.seed}.`;
  }

  async _runVideo(prompt) {
    const { motion, overridden } = motionFromQuery();
    const { blob, keyframes, meta } = await this.pipeline.generate({
      prompt,
      seconds: clampDuration(this.secondsInput.value),
      seed: this.seedInput.value.trim() || null,
      motion,
    });
    const url = URL.createObjectURL(blob);
    this.video.src = url;
    this.downloadBtn.href = url;
    this.video.hidden = false;
    this.downloadBtn.hidden = false;
    this.video.play?.().catch(() => {});
    this._renderStrip(keyframes);
    const tuning = overridden
      ? ` [${motion.ANCHOR_MODE}, anchor ${motion.ANCHOR_STRENGTH_START}->${motion.ANCHOR_STRENGTH_END}, drift ${motion.LATENT_DRIFT_START}->${motion.LATENT_DRIFT_END}]`
      : '';
    this.statusLine.textContent = `Done — ${meta.keyframes} keyframes, ${formatBytes(blob.size)}, in ${formatDuration(meta.ms)}. Seed ${meta.seed}.${tuning}`;
    this.lastResult = { blob, meta };
    this._refreshCacheInfo();
  }

  _renderStrip(keyframes) {
    if (!keyframes?.length) return;
    this.keyframeStrip.hidden = false;
    for (const kf of keyframes) {
      const c = el('canvas', { width: String(kf.width), height: String(kf.height), class: 'sm-thumb' });
      c.getContext('2d').putImageData(kf, 0, 0);
      this.keyframeStrip.append(c);
    }
  }

  /* ------------------------------- chrome ------------------------------- */

  _toggleSettings(force) {
    const open = force ?? this.settings.panel.hidden;
    this.settings.panel.hidden = !open;
    this.settings.backdrop.hidden = !open;
    if (open) this._refreshCacheInfo();
  }

  _showBanner(text, fatal) {
    this.banner.textContent = text;
    this.banner.dataset.kind = fatal ? 'fatal' : 'warn';
    this.banner.hidden = false;
  }

  _flash(text) {
    this.statusLine.textContent = text;
  }
}
