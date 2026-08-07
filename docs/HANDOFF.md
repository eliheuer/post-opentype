# Handoff: where NeuralType stands, 2026-08-04

Written so someone (or some agent) can pick this up cold. Repo root
is `~/GH/repos/post-opentype`. The blog and demo live in a separate
repo, `~/GH/repos/elih.net`. The autotracer is a third,
`~/GH/repos/img2bez`, and the workspace depends on it by path, so
both must exist side by side to build.

Machine-specific setup for the Linux training box is deliberately
not in this file. It lives in `~/Desktop/linux-training-box.md`.

## The goal right now

This work exists to support a conference talk proposal. The proposal
repo is `~/GH/repos/font-tech-ai-2026`
(github.com/eliheuer/font-tech-ai-2026), and it holds the abstracts,
the talk outlines, and the submission checklist.

| | |
| --- | --- |
| conference | Font Tech & AI, online, organized by ILT Trust |
| dates | 8-10 October 2026 |
| **proposal deadline** | **10 August 2026**, aiming to submit the 6th or 7th |
| form | https://trust.ilovetypography.com/font-tech-ai/ |

Two proposals are in play and compete with each other: `proposal.md`
is the NeuralType and nastaliq talk this repo serves, and
`proposal-virtua.md` is a Virtua Grotesk talk. `SUBMISSION.md`
explains how they are kept distinct if both go in.

Two things follow from this for anyone working in this repo. Every
number in `proposal.md` comes from the blog posts and must match what
the code actually produces, so any change to the shipping model means
updating that file. And one of the conference curators is Simon
Cozens, who is a Gulzar author, which makes the licensing and credit
handling in this project worth getting exactly right; there is an
open question in `SUBMISSION.md` about whether to name the Gulzar
authors in the abstract.

The demo target was roughly 2026-08-09, to be ready alongside the
submission. The project is paused for a day or two as of 2026-08-04
while Virtua Grotesk gets attention.

## What ships today

The live demo is the nastaliq field model, and nothing in the
experiments below beat it.

| | value |
| --- | --- |
| model | `neuraltype-field-v1`, 1,357,995 parameters |
| file | 5.4 MB, `~/GH/repos/elih.net/public/demos/neuraltype/gulzar.ntf` |
| quality | contour IoU 0.945 against the teacher |
| canvas | 155×219 signed distance field at 64 px/em |
| teacher | Gulzar Regular (OFL), shaped by harfrust |

These are the numbers `proposal.md` cites, verified against the
shipping file on 2026-08-04 and unchanged since 2026-08-02. Two
details to carry into the abstract if it is edited: the exact file
is 5,433,100 bytes, and the measured IoU is 0.945, which the blog
post and the abstract both round to 0.94.

The demo island is
`~/GH/repos/elih.net/src/components/NeuralTypeDemo.tsx`, embedded in
two posts through `NeuralTypeDemoIsland.astro` with settings from
`demo-presets.ts`. The wasm build is vendored at
`~/GH/repos/elih.net/src/lib/neuraltype-wasm/`; rebuild and copy it
after any engine change (see Commands).

## Results of every model trained so far

| model | parameters | file | outcome |
| --- | --- | --- | --- |
| field, 64 px/em | 1,357,995 | 5.4 MB | **IoU 0.945, ships** |
| field, 96 px/em | 3,101,483 | 12.4 MB | IoU 0.825, no visible gain |
| field, 64 px/em, wide | 5,357,091 | 21.4 MB | IoU 0.809, strokes merge |
| vector, outline tokens | 12,693,639 | 53.3 MB | accuracy 0.482, fragments only |

Checkpoints live in `data/train-*`. Exported fonts are in `build/`.
The 96 px and wide checkpoints were pulled back from the training
box; the vector one too.

Why each larger model lost, in short. The 96 px model spent 93% of
its parameters on one linear layer feeding a bigger deconvolution
seed grid, while the 256-number shape code never changed, and pixel
count was never the limit anyway because the tracer interpolates the
distance field to sub-pixel accuracy. The wide model was not
overfitting (training and held-out error fell together) but was
undertrained with a learning rate tuned for a model a quarter its
size, and had no normalization between the wider layers. The vector
model never learned to close a stroke: dots land in plausible
places, strokes do not form.

The long version, with figures, is the section "Three ways to make
it bigger, none of which worked" in
`~/GH/repos/elih.net/src/content/blog/nastaliq-distilled/index.mdx`.

This is also talk material. Both abstracts in
`~/GH/repos/font-tech-ai-2026/proposal.md` promise to cover "where
it currently fails", and as of 2026-08-04 that promise has real
content behind it: three scaling attempts, three different reasons
for failing, and a smallest-model-wins result.

## Next steps, ranked

1. **The data, not the model.** Every letter-in-context is
   supervised against a single shape, picked as the most common when
   the teacher draws more than one. Where Gulzar genuinely varies,
   the model learns an average of two different letters. Check how
   many contexts are shape-ambiguous (the trainer prints this at
   startup: it was 0.00% for the 64 px corpus, which is worth
   re-verifying rather than trusting) and look at whether the modal
   choice is discarding real variants.
2. **Demo polish for the talk.** The editor, the strand UI, and the
   sample sheets are the presentation material.
3. **Quantization.** f16 and int8 exports for the size table. The
   field is a smooth surface, so quantizing should be gentle, but
   that is an assumption and not yet a measurement.
4. **If revisiting the wide model**, the fixes are a lower learning
   rate, normalization between layers, and enough epochs to
   converge. Not obviously worth it before the talk.
5. **If revisiting the vector model**, the suspects are the
   tokenization (delta chains make exact-token prediction noisy) and
   the objective (no learning rate schedule, no warmup). Its
   engine-side support is already written and working, so only the
   training question is open.
6. **Richer field representations** (untested ideas, 2026-08-07).
   Three ways to pack more shape into fewer parameters, all from the
   same graphics-trick family as the SDF:
   - **Multi-channel SDF (MSDF, Chlumsky's msdfgen technique).**
     Three overlapping distance fields in RGB, median-decoded,
     reconstruct sharp corners from much smaller grids. Attacks a
     measured cost: in the 96 px/em run, 93% of parameters sat in
     the one linear layer that seeds the pixel grid. If corners
     survive at lower resolution, the canvas and that layer shrink.
     Caveat: nastaliq is mostly smooth curves, so the win may be
     modest; square Kufic or a geometric Latin would benefit most.
   - **Distance plus gradient.** Predict the field's gradient
     (edge direction) alongside the distance, per cell. Edge
     position plus direction lets a tracer reconstruct curves from
     coarser grids; this is the Hermite-data idea behind dual
     contouring. Same goal as MSDF by a different route.
   - **Coordinate network (DeepSDF / SIREN style).** Drop the grid:
     the network takes (x, y, letter context) and returns one
     distance. No deconv stack, no seed grid, no resolution
     parameter at all; the tracer samples wherever it wants at any
     precision, and every parameter describes shape rather than
     pixels. The scaling post-mortem (pixels were never the
     bottleneck, the seed grid was dead weight) points here. Also
     the strongest "where this goes next" beat for the talk: the
     font becomes a pure function from position and context to ink.

## Repo map

Crates:

- `crates/neuraltype-core` is the engine: `field_model.rs`
  (hand-rolled field inference), `vector_model.rs` (hand-rolled
  transformer inference with a KV cache), `field_text.rs` (clusters,
  the displacement cascade, word compositing, the fast tracer),
  `trace.rs`, plus the older Kufic model.
- `crates/neuraltype-wasm` is the browser binding. `shape()` returns
  JSON the demo island draws. It never traces with img2bez inline:
  the worker calls `trace_word_svg`, the main thread installs the
  result with `insert_word_trace`.
- `crates/neuraltype-train` has two binaries. `ntf-train` (`main.rs`)
  trains the field model and `export.rs` writes the .ntf.
  `ntf-train-vec` (`bin/vec.rs`) trains, samples, exports and renders
  the vector model.
- `crates/neuraltype-distill` extracts the teacher, builds fields,
  and renders proof sheets.
- `crates/ntf-dash` is the training dashboard (ratatui).
- `vendor/candle-kernels` is patched to build on older GPUs; see the
  desktop runbook for why.

Data:

- `data/extract-gulzar/` teacher outlines and cluster records.
- `data/fields-gulzar-64/`, `data/fields-gulzar-96/` rendered SDFs.
- `data/vec-gulzar/` tokenized outline sequences for the vector
  track, built by `scripts/vec_tokenizer.py`.

Docs: `docs/SPEC.md` (format), `docs/DISTILL.md` (the plan this
implements), `docs/VECTOR.md` (vector track), `docs/COMPRESSION.md`
(size work), `docs/NASKH.md` (direction).

## Commands

Train the field model (add `--features cuda` on a GPU box, or
`--features accelerate` on the Mac; Metal is about twenty times
slower than CPU and should not be used):

```sh
cargo build --release -p neuraltype-train --features accelerate --bin ntf-train
target/release/ntf-train <fields-dir> <out-dir> <epochs>
```

Environment knobs for that trainer: `NTF_LR` (default 3e-4),
`NTF_BS` (128), `NTF_OVERSAMPLE` (repeats long-word rows, which is
how the الله ligature was forced to converge; K=48 fixed it in eight
epochs, K=8 is maintenance), `NTF_LATENT` (256), `NTF_CHANS`
(`128,64,32,16,8,1`), `NTF_EMB` (24). The epochs argument means
additional epochs: the trainer resumes `checkpoint.safetensors` from
the output directory if it exists.

Export a field font. Architecture is read back from the checkpoint's
tensor shapes, so a run trained with custom dimensions exports
correctly without repeating its environment:

```sh
target/release/ntf-train export <train-dir> <fields-dir> <style-name> <out.ntf>
```

Vector track:

```sh
python3 scripts/vec_tokenizer.py            # writes data/vec-gulzar
target/release/ntf-train-vec <vec-dir> <out-dir> <epochs>
target/release/ntf-train-vec export <out-dir> <vec-dir> <extract-dir> <out.ntf>
target/release/ntf-train-vec render <ntf> <word> [out.svg]
```

Its dimension knobs are `NTF_D` (384), `NTF_LAYERS` (7), `NTF_HEADS`
(8), `NTF_FFN` (1536), which together give the 12.7M configuration.
`NTF_BS` must be small (8) because attention over ~400 tokens is
memory hungry.

Comparison sheets, the honest quality gate. Both binaries expose
`wordjson`, and the script puts teacher, model B and model C in one
coordinate frame:

```sh
SHEET_TITLE="..." SHEET_B="..." SHEET_C="..." \
  python3 scripts/sample_sheet.py <b.ntf> <c.ntf> <out.html> [words...]
```

`SHEET_BARE=1` drops the header and stat cards, for making figures
to embed in a post. Render one to PNG with headless Chrome:

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --screenshot=out.png \
  --window-size=1240,700 --force-device-scale-factor=2 file://<path>
```

Rebuild the engine for the demo:

```sh
cd crates/neuraltype-wasm && wasm-pack build --release --target web
cp pkg/neuraltype_wasm.js pkg/neuraltype_wasm_bg.wasm pkg/*.d.ts \
   ~/GH/repos/elih.net/src/lib/neuraltype-wasm/
```

Watch a training run (needs a real terminal, and the second argument
is the target epoch count, which the log itself does not know):

```sh
target/release/ntf-dash <log-file> <target-epochs>
```

## Sharp edges

- **The trainer overwrites its checkpoint every epoch.** The file on
  disk is the latest epoch, not the best one. IoU wobbles between
  epochs, so a run that ends on a bad epoch exports a worse model
  than it trained. This has not bitten us because decay legs end on
  settled curves, but it is real.
- **IoU is measured on the model's own pixel grid**, so it cannot be
  compared between models trained at different canvas resolutions.
  This is why 0.945 at 64 px/em and 0.825 at 96 px/em are not a
  like-for-like comparison.
- **Metrics hide ligatures.** A single wrong ligature is invisible
  when averaged over 91,339 contexts. The 96 px model scored
  respectably while drawing الله wrong. Always look at a sample
  sheet before believing a number.
- **Never pipe a build through `tail` when checking success.** The
  pipe swallows the exit code, and a failed build looks like a
  successful one until the binary turns out to be missing.
- **Vite serves stale caches** after engine changes. Stop the dev
  server, `rm -rf ~/GH/repos/elih.net/node_modules/.vite`, restart.
- **The vector .ntf is 53 MB** and was deliberately kept out of the
  site repo. The demo supports it: set `vectorFont` in
  `demo-presets.ts` to a `neuraltype-vector-v1` file and a model
  picker appears in the UI. The exported file is
  `build/gulzar-vec-120.ntf`.
- **Commit locally, push only when asked.** That is the standing
  rule on this project.
