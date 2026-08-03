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

## Sequencing

1. After the 96 px/em run ships: tokenize the Gulzar cluster
   outlines (quantized coords; reuse the Virtua vocabulary design).
2. Prototype the decoder on the kiln; evaluate with the same
   compare gates (teacher sheets, basmala, IoU after rasterizing).
3. Compare the three columns honestly: field-64, field-96,
   vector-v2 — fidelity, file size, inference speed per keystroke.
4. If vector wins, it becomes "neuraltype-vector-v2" in the spec and
   the blog gets its third post.
