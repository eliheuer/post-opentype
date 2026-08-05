# Vector-native .ntf research

Goal: a .ntf where the model outputs bezier outlines directly, with
no field and no tracer. The Virtua-12M work (elih.net, virtua
grotesk post) is the template: a small decoder-only transformer that
emits drawing commands token by token, trained on one machine.

## Why this is tractable now

The extraction stage already produced the supervision. For every
cluster in the corpus, `data/extract-gulzar/glyphs.jsonl` holds the
teacher's own bezier paths. Autoregressive training with teacher
forcing on those canonical sequences removes the point-correspondence
problem that usually blocks vector-output models. The loss is
next-token cross-entropy. Greedy decoding keeps inference
deterministic, so the format's "same text, same outlines" claim
holds.

## Sketch of v2

- Context: the same 5-token window (prev2, prev, letter, next,
  next2), embedded as today.
- Decoder: a small transformer (start at 2-6M params) that emits
  quantized (command, dx, dy) tokens for the cluster outline.
  Vocabulary like Virtua-12M: commands, coordinates, deltas.
- Displacement head: unchanged. The chain and the cascade are
  representation-agnostic.
- Training: multi-task. Keep the field head as an auxiliary loss on
  a shared trunk; it stabilizes training with dense gradients. Ship
  only the vector head in the font file.

## What it buys

1. Resolution becomes a non-problem. No 64 vs 96 px/em ceiling and
   no retraining to sharpen.
2. File size collapses. Outline tokens are bytes per shape; the
   megapixel decoder (2.9M params of l2 alone at 96 px/em)
   disappears. This rewrites docs/COMPRESSION.md from below.
3. Editor-native output: real on-curve/off-curve structure, straight
   into Runebender. No tracing pass.
4. The shared-prior/LoRA direction gets stronger: the prior and the
   font deltas live in the same architecture family as Virtua-12M.

## Risks and mitigations

- Thin features and dots are where autoregressive outline models
  glitch (spikes, dropped points). Mitigation: the graded-corpus
  discipline from Virtua (human-graded examples), plus the field
  auxiliary loss.
- The continuous-stroke claim: teacher outlines are per-glyph and
  join by overlap. Options: path boolean union, or rasterize the
  outlines and round-trip through img2bez::trace_sdf for display.
- Editing UX: selection clouds and junction nodes consume the SDF.
  Vectors-to-field is the cheap direction (rasterize + exact EDT on
  demand), so the strand system survives unchanged.

## Sequencing: two tracks in parallel

Decision (2026-08-03): do not pick a direction. The field track and
the vector track train in parallel and race inside one harness. The
.ntf container makes this cheap: the header's `format` field already
dispatches per-file in the engine, so a "neuraltype-vector-v1" font
loads in the same demo, drives the same strand UI, and passes the
same gates as the field fonts.

1. Field track: the 96 px/em run and its fine-tune legs continue on
   the kiln, unchanged.
2. Vector track, build order:
   a. Tokenizer: Gulzar cluster outlines from
      data/extract-gulzar/glyphs.jsonl into quantized command
      sequences (reuse the Virtua vocabulary design).
   b. Trainer: a small candle transformer next to the field trainer.
      Virtua-scale models train in hours on the Mac or the kiln.
   c. Engine: a "neuraltype-vector-v1" loader arm in the wasm
      dispatch; selection clouds come from rasterize + exact EDT.
3. One comparison table, same gates for every column: field-64,
   field-96, vector-v1 — fidelity (teacher sheets, basmala), file
   size, inference speed per keystroke.
4. The winner earns the spec name and the blog's third post; the
   loser still informs the shared-prior work in COMPRESSION.md.

## Result (2026-08-04): the field track wins for now

All of stage 2 is built and working. The tokenizer produced 91,339
sequences (135-token vocabulary, 17M tokens, mean length 187). The
trainer is `crates/neuraltype-train/src/bin/vec.rs`. The engine arm
is `crates/neuraltype-core/src/vector_model.rs`: hand-rolled
decoder inference with a per-layer KV cache, greedy decoding, and
the detokenizer back to béziers. Placement comes from a per-context
displacement table in the file, so the cascade chains exactly as it
does for field fonts. `neuraltype-wasm` loads the format, and the
demo grows a model picker when `vectorFont` is set.

What did not work is the model. A 12,693,639-parameter decoder
(d=384, 7 layers, 8 heads, ffn 1536) trained 120 epochs plateaued at
0.482 next-token accuracy. Two learning-rate drops (3e-4 to 1e-4)
moved it by 0.002. Decoded output is a scatter of small closed
contours: dots land near where dots belong, and on longer words the
fragments follow the downward slope of the baseline, so the model
learned something about placement. It never learned to close a
stroke.

Since token accuracy punishes an off-by-one coordinate bin as hard
as a wrong command, the number alone was not proof; the sample sheet
was. Build one with `scripts/sample_sheet.py` before drawing any
conclusion from a metric.

Open suspects, if this is picked up again: the delta-chaining
tokenization may make exact-token prediction inherently noisy, and
the training objective has no warmup and no schedule. The engine
side needs no further work.
