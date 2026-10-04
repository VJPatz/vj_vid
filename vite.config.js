import { defineConfig } from 'vite';

// Static single-origin app, no SSR/API — serves from dist/ on any host.
// No COOP/COEP: would unlock multi-threaded WASM but breaks cross-origin
// weight loading from HF. ORT is pinned single-threaded instead (ort.js).
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 4096, // ORT wasm + tokenizer are large by design
    rollupOptions: {
      output: {
        manualChunks: {
          ort: ['onnxruntime-web'],
          transformers: ['@huggingface/transformers'],
        },
      },
    },
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'], // ships prebuilt wasm; let the runtime loader control paths
  },
  server: {
    port: 5173,
    host: true,
  },
});
