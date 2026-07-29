# NeuralType `.ntf` — a post-OpenType generative font format

**Status: v0 draft, implemented.** This document specifies what a
`.ntf` font is, how an engine renders text with one, and the roadmap
from the current square-Kufic v0 to naskh and nastaliq.

## 1. Premise

A font is a function, not a lookup table:

```
glyph = F(character, context, parameters)
```

OpenType approximates `F` with a finite table of outlines plus a rule
system (cmap, GSUB, GPOS) selecting among them. For connected scripts
the approximation is coarse: Arabic gets four contextual classes
(isolated/initial/medial/final) where the written tradition has a
continuum; justification by kashida is faked with a separate tatweel
glyph; nastaliq's context-dependent stacking defeats the model almost
entirely.

A `.ntf` font stores `F` directly, as the weights of a small neural
network. The engine evaluates `F` per glyph, per keystroke, and traces
the result into bezier outlines. Everything OpenType compiles into
tables — contextual forms, elongation, (eventually) optical
corrections and stylistic variation — becomes behaviour of one learned
function with continuous inputs.

Consequences worth naming:

- **Continuous parameters.** Elongation (kashida) is a scalar input,
  not a glyph. Any learned axis (weight, slope, formality) works the
  same way — variable fonts fall out for free.
- **Open-ended context.** The conditioning vector can include
  neighbouring letters, position in word/line, or justification
  pressure. Contextual forms are not enumerated; they are generated.
- **The file is small and uniform.** No table graph, no rule bytecode:
  a header and a weight blob.

## 2. File format (v0, `neuraltype-mlp-v0`)

```
offset  size    content
0       4       magic "NTF0"
4       4       u32 LE: header length H
8       H       JSON header
8+H     ...     raw little-endian f32 weights, layer by layer
                (each layer: W row-major [n_out × n_in], then b)
```

JSON header fields:

| field      | meaning                                            |
|------------|----------------------------------------------------|
| `format`   | `"neuraltype-mlp-v0"`                              |
| `script`   | `"arabic"`                                         |
| `style`    | e.g. `"square-kufic"`                              |
| `alphabet` | string of supported codepoints, index = model id   |
| `layers`   | layer sizes, e.g. `[35, 128, 128, 225]`            |

Hidden layers are ReLU; the output layer is linear. (Quantization to
f16/int8 is an obvious follow-up; v0 keeps f32 for simplicity.)

## 3. The v0 model (square Kufic)

Square Kufic is the deliberate first rung: its letterforms live on a
coarse binary grid (strokes 1 cell wide, counters 1 cell), so the
generative representation is trivial and every part of the pipeline is
inspectable.

**Input** (35 floats):
- one-hot letter identity over the alphabet (30) — note dots are
  *generated*, not input: ب/ت/ث are distinct ids whose shared rasm the
  model discovers in training;
- one-hot joining form (4): isolated / initial / medial / final;
- elongation scalar ∈ [0,1]: kashida columns / MAX_ELONG.

**Output** (225 floats):
- 16×14 occupancy grid (threshold at 0.5), canvas right-aligned at the
  pen position, baseline at row 10, descender zone below;
- advance width / 16.

**Engine pipeline** (`neuraltype-core`):
1. *Shape*: Unicode text → joining forms (standard Arabic joining
   rules; right-joiners د ذ ر ز و ا have no initial/medial forms).
2. *Generate*: one forward pass per glyph → occupancy grid + advance.
3. *Trace*: exposed cell edges, oriented filled-side-left, chained
   into closed loops → rectilinear `kurbo::BezPath` (outer contours
   and holes get opposite windings; nonzero fill).
4. *Layout*: RTL pen, connected letters abut (the model draws its own
   connectors), 1-cell gap after non-connecting letters, 3-cell word
   space.

**Training** is distillation from a procedural teacher: letterforms
authored as ASCII-art grids (exactly a square-Kufic chart), composed
with dots and elongation. Full-batch Adam, MSE, ~4k epochs, seconds on
CPU. The teacher never ships; the `.ntf` weights are the font.

## 4. Roadmap

### v1 — naskh: from occupancy to strokes

Swap the output head; keep everything else.

- **Output**: a fixed-size skeleton — up to N anchor points, each
  `(present, new-stroke, x, y, in-handle, out-handle, pen-width)` —
  plus dot positions and advance. Outlines come from kurbo stroke
  expansion of the generated centerline with variable width, i.e. a
  learned broad-nib pen. This is Metafont's insight (letters are pen
  strokes over a skeleton) with the parameter logic *learned* instead
  of hand-programmed.
- **Conditioning grows**: neighbouring letter identities (prev/next),
  position in word. This is where the format starts doing what
  OpenType cannot: بـ before ح takes a different tooth than بـ before
  ي, without anyone enumerating the cases.
- **Teacher**: a parametric naskh skeleton engine on a coarse grid
  (the same authoring discipline as v0), or fitting to existing
  outlines/manuscript scans.

### v2 — nastaliq

The target the format exists for. Additional conditioning: position
along the sloped connection cascade (letters in a nastaliq word step
diagonally down), accumulated stack height, line-fitting pressure.
Output gains a 2D pen offset per glyph so the model places letters
above/left of the pen, not just along a baseline. Success criterion:
a word like نستعليق rendered as one descending gesture, its shape a
function of the whole word — which is simply what nastaliq *is*.

### Open questions

- **Determinism**: rendering must be reproducible across engines →
  spec pins the activation functions, evaluation order, and threshold
  semantics; weights are exact bytes. (No sampling — `F` is a
  deterministic function; "generative" here means *generates shapes*,
  not *stochastic*.)
- **Hinting/quality floor**: how to guarantee legibility for inputs
  off the training distribution (rare words are fine — the context
  vector, not the word, is the input — but future free-form
  conditioning needs guardrails).
- **Text stack integration**: `.ntf` replaces glyph storage *and*
  GSUB/GPOS, but selection, cursors, and accessibility need cluster
  mapping — the engine must report text↔outline correspondence (v0
  returns per-glyph paths tagged with source character and form).
- **Size/speed budget**: v0 is 200 KB / ~50k params / trivially
  realtime. A naskh model will be bigger; the interesting constraint
  is staying under typical webfont sizes (~100–500 KB) — early
  evidence says yes with quantization.
