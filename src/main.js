/** Entry point — mounts the UI. Real work happens lazily in the pipeline on Generate. */
import { App } from './ui/app.js';
import { logger } from './core/log.js';

logger.info('VJ_VID booting');

const root = document.getElementById('app');
try {
  // eslint-disable-next-line no-new
  new App(root);
} catch (err) {
  logger.error('boot failed', { error: String(err?.message ?? err) });
  root.innerHTML = `<div class="sm-banner" data-kind="fatal">VJ_VID failed to start: ${String(err?.message ?? err)}. Reload the page.</div>`;
}
