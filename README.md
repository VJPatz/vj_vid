<div align="center">

# vj_vid

**Edge AI image & video generation — two diffusion/vision models running
entirely client-side, in a browser tab, with zero backend.**

No server. No API keys. No inference cost. Every forward pass — text
encoding, denoising, decoding, optical-flow interpolation, H.264 encoding —
executes on the user's own GPU via WebGPU, or falls back to WASM.

[**Try it live**](#) · [Watch on YouTube](#) · [Architecture](#architecture) · [Why edge AI](#why-edge-ai) · [Contributing](#contributing) · [Disclaimer](#disclaimer)

</div>

---

## What this is

Type a prompt. Get an image, or a short mp4, generated on your own machine —
nothing leaves the browser after the page first loads.

```
prompt → SD-Turbo keyframes (WebGPU) → RIFE frame interpolation → mp4 (WebCodecs)
```

Two separate neural networks run in sequence, both entirely on-device:

1. **SD-Turbo** (distilled Stable Diffusion) turns text into image keyframes.
2. **RIFE** (optical-flow frame interpolation) turns those keyframes into
   smooth motion, synthesized frame-by-frame between two real model outputs.

~2.5 GB of model weights download once and cache on-device. Every run after
that is fully **offline**.

## Try it live

**[→ Open the deployed app](#)** — no install, nothing to run locally.

First load downloads ~2.5 GB of model weights once (progress shown live,
with a percentage and bytes downloaded/remaining). After that, weights are
cached in your browser (OPFS) and every generation after the first is
instant-start.

If you find this useful or interesting, a ⭐ on the repo helps it get seen —
and PRs/forks are genuinely welcome, see [Contributing](#contributing).

## Demo

[![Watch the demo](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

*Click to watch on YouTube — walkthrough of image/video generation running
entirely in-browser, no server round-trip.*

## Why edge AI

Running inference in the browser is a fundamentally different engineering
problem than calling an API, and that's the point of this project:

- **No server means no server-side safety net.** Memory budgets, execution
  provider availability, download reliability, browser sandbox limits — all
  of it is the application's problem, not an ops team's.
- **Two unrelated model architectures, one runtime.** A latent diffusion
  UNet and an optical-flow CNN have different input conventions, different
  tensor shapes, different numeric ranges — both had to run inside the same
  `onnxruntime-web` session-management layer (`src/engine/ort.js`) without
  either one leaking its assumptions into the other.
- **Failure modes an API never exposes.** A model repo that only worked on
  CUDA and silently rejected every WebGPU/WASM session. A WebGPU backend
  that never registered because of a single wrong import path — the app
  looked like it was trying WebGPU and falling back, when it was never
  requesting WebGPU at all. A VAE decoder throwing a float16/float32
  mismatch traced to a specific `onnxruntime-web` version's WebGPU execution
  provider, not the model. These are the bugs you get building a runtime,
  not calling one.
- **Coherent motion from independent samples, with no video model.** There
  is no text-to-video model here — RIFE is an interpolation model, not a
  generator. Getting diffusion keyframes to read as one continuous scene
  instead of a slideshow required designing a shared-noise latent drift +
  anchor-blend scheme from scratch (`src/engine/motion.js`), so RIFE always
  has two frames with enough underlying structure in common to interpolate
  cleanly between.

## The two models

### SD-Turbo — image generation

A distilled, single-step version of Stable Diffusion 2.1. Where a normal
diffusion model needs 20–50 UNet passes to denoise an image, SD-Turbo is
trained to do it in one — which is what makes browser-side generation
tractable at all. Each keyframe here is exactly one UNet evaluation.

Pipeline: CLIP text encoder → UNet (single denoising step) → VAE decoder →
RGBA. All three graphs run via `onnxruntime-web`, loaded and released one
phase at a time so peak memory stays near the largest single graph (~1.9 GB)
instead of the sum of all three (~2.5 GB).

### RIFE — motion, without a video model

RIFE (Real-Time Intermediate Flow Estimation) is an optical-flow model: given
two images, it estimates how pixels moved between them and synthesizes a
genuine in-between frame — not a crossfade, not a camera pan over static
images, an actual interpolated frame built from estimated motion.

This project uses it to turn independently-generated SD-Turbo keyframes into
video. Because RIFE only produces the exact midpoint of a given pair, hitting
an arbitrary frame count means recursively bisecting the gap between each
pair of keyframes — the standard RIFE multi-frame recipe, implemented in
`src/engine/video.js`.

RIFE will not invent motion the keyframes don't already imply — it
interpolates what's actually there, which is why the keyframes need to be
visually close to begin with (see the anchor-blend design below).

## What's borrowed vs. what's built

To be direct about it: neither model's weights are mine.

| | |
|---|---|
| **Not mine** | SD-Turbo weights ([Stability AI](https://huggingface.co/stabilityai/sd-turbo)), browser-ready ONNX export ([schmuell/sd-turbo-ort-web](https://huggingface.co/schmuell/sd-turbo-ort-web)), RIFE v4.9 weights ([edgetools/rife](https://huggingface.co/edgetools/rife)) |
| **Mine** | The browser inference runtime, the WebGPU→WASM execution pipeline shared across both models, the resumable weight-caching layer, the memory-budgeted session lifecycle, the latent-motion coherence system, the RIFE-based video assembly pipeline, and the debugging that got two different model classes to actually run reliably client-side |

This is the engineering built *around* pretrained models — the relationship
essentially all applied-ML work has to foundation models. The skill on
display isn't training SD-Turbo or RIFE; it's making two independently-built
models run fast and reliably together under constraints an API call never
exposes: no server, no control over the client's GPU, no control over the
network.

## Contributing

Issues, forks, and PRs welcome — this is an active playground, not a
closed/finished artifact. Useful starting points:

- `src/engine/motion.js` / `registry.js` — the latent-motion coherence
  system (anchor/drift tuning) is the most actively-evolving part of this
  project; ideas on better coherence-vs-motion tradeoffs are especially welcome.
- `src/engine/video.js` — RIFE interpolation and frame-tree assembly.
- `npm run verify` before opening a PR — static import check + unit tests,
  no GPU required.

If this is useful to you, starring the repo helps others find it.

## Quick start

```bash
npm install     # also copies the ONNX runtime wasm into public/ort/
npm run dev      # http://localhost:5173
```

Open in **desktop Chrome or Edge 121+** (WebGPU + WebCodecs required). First
generation downloads ~2.5 GB of model weights — cached in OPFS afterward,
and resumable if interrupted. Every run after that skips the download.

```bash
npm run verify   # static check + unit tests — runs anywhere, no GPU/network
npm run build    # static production bundle in dist/
npm run preview  # serve the built bundle
```

## Features

- **Image mode** — single prompt → single image.
- **Video mode** — multiple keyframes, 2–20s, 30fps, 512×512 → mp4.
- **Deterministic generation** — same prompt + seed supposedly reproduces identical
  output, on any machine.
- **Fully offline after first load** — resumable, chunked downloads into
  OPFS; no re-fetch on reload.
- **Live progress** — percentage + bytes downloaded/total on first run,
  per-keyframe progress during generation, for every phase of the pipeline.

## Architecture

```
UI (src/ui/app.js)
   │
   ▼
Pipeline (src/engine/pipeline.js)
   │
   ├─ DOWNLOAD       resumable weight fetch, OPFS/Cache/memory cache
   ├─ ENCODE_TEXT     CLIP tokenize → text encoder (WASM)
   ├─ KEYFRAMES       shared drifting noise → UNet (WebGPU, 1 step/frame)
   ├─ DECODE          VAE decode → RGBA
   ├─ RIFE_DOWNLOAD   RIFE weights (~22 MB)
   └─ ENCODE_VIDEO    RIFE optical-flow interpolation → WebCodecs → mp4
```

Each model (text encoder, UNet, VAE decoder, RIFE) is loaded for its phase
and released before the next, so peak memory stays near the single largest
graph instead of the sum of all of them.

**Keyframes are sampled independently, not chained.** Each is one SD-Turbo
step from a shared base noise field that slowly drifts across the clip
(camera motion), with every frame's prediction anchor-blended toward the
*previous* frame's prediction to hold scene identity — a chain, not a
hub-and-spoke, so motion compounds forward instead of every frame reading as
"frame 0 with distortion." Anchor strength and drift amount are ramped
across the clip (strong lock at the start, loosening by the end), tunable
without a rebuild via URL params. This sidesteps the drift/collapse that
naive img2img chaining causes on a distilled single-step model — and it's
what gives RIFE two frames coherent enough to interpolate between cleanly.

### Project layout

```
src/
  core/       errors, logging, seeded RNG, event emitter  (framework-free)
  engine/
    capability.js  WebGPU / WebCodecs / storage probe
    cache.js       resumable OPFS / Cache API / memory weight cache
    ort.js         shared ONNX Runtime loader (WebGPU → WASM fallback)
    registry.js    pinned model config + product limits
    tokenizer.js   CLIP tokenizer
    scheduler.js   diffusion math — pure, unit-tested
    motion.js      latent drift / slerp / anchor blend — pure, unit-tested
    prompt.js      prompt scheduling — pure, unit-tested
    sdturbo.js     SD-Turbo ONNX graphs + per-phase memory discipline
    rife.js        RIFE wrapper
    video.js       RIFE interpolation → WebCodecs mp4 assembly
    pipeline.js    orchestrator: prompt → image / video
  ui/         framework-free DOM controller
  styles/     one stylesheet, light + dark
scripts/      postinstall ORT asset copy, static import/syntax check
test/         unit tests (node --test)
```

## Model details

| | |
|---|---|
| Keyframes | SD-Turbo (SD2.1-distilled, single-step), fp32 ONNX graph I/O |
| Text encoder | OpenCLIP-ViT-H, hidden size 1024, 77-token context |
| Tokenizer | `openai/clip-vit-large-patch14` (byte-identical CLIP BPE vocab) |
| VAE | scale 0.18215, 4-channel 64×64 latents, decoder only |
| Motion | RIFE v4.9, optical-flow frame interpolation, ~22 MB |
| Runtime | onnxruntime-web 1.17.1 (exact pin), WebGPU with WASM fallback, single-threaded |

## Testing

```bash
npm run verify
```

Runs a static import/syntax gate (every module parses, every relative import
resolves to a real export — catches the renamed-symbol class of bug before
it ever reaches a browser) plus unit tests covering the diffusion scheduler
math, latent motion, prompt scheduling, and RIFE frame-tree interpolation —
all pure functions, testable without a GPU.

## Deploying

```bash
npm run build
```

Outputs a static bundle to `dist/`. Host it anywhere — no special headers
required. A GitHub Actions workflow (`.github/workflows/deploy.yml`) deploys
to GitHub Pages on push to `main`.

## Disclaimer

This project is a technical demonstration of client-side generative AI. It
provides no content moderation, filtering, or output review of any kind —
by design, since there is no server in the loop to add one.

Outputs are generated entirely by third-party models (SD-Turbo, RIFE)
running on the end user's own device, from whatever prompt they supply. The
author(s) of this repository do not control, endorse, monitor, or take
responsibility for any content generated through it. Users are solely
responsible for the prompts they submit and the outputs they produce, share,
or use, and for ensuring that use complies with applicable law and with the
underlying models' own licenses and usage policies (see [License](#license)).

This software is provided "as is," without warranty of any kind — see
[LICENSE](LICENSE).

## License

App code is MIT — see [LICENSE](LICENSE). The AI models it downloads at
runtime carry their own licenses (SD-Turbo is non-commercial; RIFE is MIT) —
see LICENSE for details. You are responsible for complying with those terms
when running the app.
