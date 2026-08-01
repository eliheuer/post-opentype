# Compression research

Goal: a .ntf font that is not more than 2x the size of a large
OpenType font for the same script. Stretch goal: smaller than the
OpenType font. This document collects the research directions, the
measurements, and the results.

Reference numbers, 2026-07-31:

| artifact | size |
|---|---|
| Gulzar-Regular.ttf | 963 KB |
| gulzar .ntf, f32, triple corpus | 5.4 MB (1,357,971 params) |
| estimate for full letter-sequence coverage, f32 | 12–24 MB (3–6M params) |

The parameter count scales with the shape inventory and the rule
window, not with the number of possible words. The model computes a
local function: five context tokens map to one shape and one
displacement. A longer word is more applications of the same
function, not a larger function.

## 1. Quantization

Expected gain: 4x to 8x. Effort: low. Status: not started.

- f16: 2x, no measurable loss expected. Make this the default
  interchange encoding.
- int8: 4x. SDF regression tolerates weight noise because the output
  is thresholded at zero. Measure contour IoU against the teacher
  before and after.
- int4 with quantization-aware fine-tuning: 8x. Run the last training
  epochs with fake-quantized weights so the model adapts.
- Per-tensor vs per-channel scales: measure both.

At int8, the current 5.4 MB file becomes approximately 1.4 MB. That
is under 1.5x Gulzar today. The open question is whether the ratio
holds when the corpus grows to full coverage.

## 2. Architecture

Expected gain: 2x to 5x at equal fidelity. Effort: medium. Status:
not started.

The generic deconvolution stack pays for the ability to draw any
image. The shape space is much more structured: approximately 400
base glyphs, composed with dots and marks, then displaced. Ideas:

- Factored decoder: a compact glyph decoder plus a small
  context-to-composition head. The context head selects and places;
  the decoder draws.
- Low-rank factorization (SVD) of the dense layers. The l2 layer
  (256 -> 4480) holds most of the MLP parameters.
- Magnitude pruning plus sparse storage. The ReLU layers are already
  activation-sparse; measure weight sparsity after training.
- Depthwise-separable deconvolutions in the decoder.

## 3. Shared prior (base model plus font delta)

Expected gain: 10x or more per font. Effort: high. Status: idea.

Ship one large letterform prior once, like a rasterizer or a system
library. Each font is then a small delta against the prior:

- LoRA-style adapters on the prior's weights.
- A codebook residual: the prior supplies the codebook, the font
  supplies indices and small corrections.

The prior amortizes across every font. A per-font file could then be
smaller than its OpenType equivalent. Open questions:

- Does the existing virtua-12m model work as a prior? It must learn
  Arabic letterform structure before it can help nastaliq. Check what
  its training data covered.
- Spec question for .ntf v2: how does a font file reference a
  standard base model, and how are prior versions pinned?

## 4. Container

Expected gain: 1.2x to 2x. Effort: low. Status: not started.

- Entropy-code the quantized weights inside the container (zstd or
  range coding over int8 values).
- Weight clustering (palettization): store a small palette per tensor
  and one index per weight.
- Keep the header JSON; it is under 2 KB and self-describing.

## Measurement protocol

For each experiment, record:

1. File size in bytes.
2. Parameter count and encoding.
3. Contour IoU against the teacher on the held-out set.
4. A proof sheet of the worst 20 words, judged by eye.
5. Decode speed in the WASM engine (ms per glyph on one core).

A result counts when the contours pass visual inspection at reading
size, not only when the IoU is high.

## Sequencing

1. After the corpus-v2 fine-tune: export f16 and int8, measure, and
   publish the size table in the blog post.
2. Then pruning and low-rank factorization on the current
   architecture.
3. The factored decoder and the shared prior are corpus-v3 era
   research.
