# v1 Research: Naskh and Nastaliq

**Status: research, not implemented.** This document works out how to
build the naskh demo, and after it the nastaliq demo. It is meant to be
argued with.

## 1. What v0 proved, and what it dodged

Square Kufic collapsed every hard problem into an easy one:

| Problem | v0 answer | Why it was easy |
| --- | --- | --- |
| Shape representation | 16×14 binary occupancy grid | fixed size, binary, advance derivable |
| Training data | 437 hand-authored ASCII grids | enumerable by one person in one file |
| Quality metric | cell-exact reproduction | no such thing as "almost right" |
| Letter joining | merge bitmaps on one baseline row | compositing does the work |

Naskh reopens all four. The letterforms are smooth curves with a
modulated stroke. The contextual variation is larger than four forms.
"Exact" stops being the metric because the outputs are continuous. And
the joins are curved transitions in a joining zone, not a shared row of
cells. Nastaliq adds a fifth problem: letters stack down a diagonal
cascade whose geometry depends on the whole word.

The rest of this document takes the four problems in order of how much
they constrain everything else: representation, data, joining, quality.

## 2. Output representation

Four candidates.

### a. Signed-distance field (the continuous version of v0)

The model outputs a grayscale distance field on a finer grid (64×56 or
96×84 instead of 16×14 binary). The engine composites fields per line
(pointwise max), takes the zero iso-contour, and fits curves to it.

- Pro: it is v0's pipeline with one type changed. Compositing still
  solves joining. Every piece of the current engine survives.
- Pro: the tracing stage is a solved problem in this ecosystem:
  [img2bez](https://github.com/eliheuer/img2bez) exists precisely to
  turn a raster into a font-quality bézier outline with points at
  extrema and corners. Feed it the composited line, supersampled.
- Pro: training data is nearly free (section 3).
- Con: resolution-limited. Hairline details need field resolution.
- Con: the outline point structure is procedural (img2bez's), not
  learned. For a demo this is fine; it is img2bez's whole thesis.

### b. Skeleton + pen (the Metafont lineage)

The model outputs a centerline: a short sequence of on-curve points
with tangent handles and a pen width (and possibly pen angle) per
point. The engine expands the stroke into a filled outline.

- Pro: matches what naskh is. The script is pen strokes over a
  skeleton; Thomas Milo's analysis and the whole Metafont/AlQalam
  lineage model it this way.
- Pro: radically compact. A naskh letter body is 3 to 12 skeleton
  points, roughly 100 output floats per glyph, smaller than v0's 224.
  The model stays under ~100k parameters. This representation matches
  the intrinsic dimensionality of the script.
- Pro: joining becomes geometry (section 4): weld skeleton endpoints.
- Pro: this is the representation the editor (in-context skeleton
  drawing, corrections as training data) wants.
- Con: variable-width stroke expansion is real engineering. kurbo has
  constant-width stroking and cubic offset curves; variable width means
  offsetting with a varying distance and fitting. Prior art:
  Kalliculator, FontForge's Expand Stroke, RoboFont outliner, and Raph
  Levien's parallel-curve work. Tractable, not trivial.
- Con: a pure pen model does not capture everything. Naskh terminals
  (tapered tails, head serifs, the eyes of م و ف) need pen-angle
  changes or explicit terminal geometry. Metafont handled this with
  elaborate pen machinery; we would start simpler and accept a plainer
  naskh.
- Con: training data requires skeletons, which nobody publishes
  (section 3).

### c. Direct outline points

The model outputs the final outline as a fixed-maximum sequence of
cubic segments with validity flags (the DeepVecFont-style head).

- Pro: captures any shape, including terminals, exactly.
- Con: highest dimensionality; self-intersection and winding errors
  are possible outputs; and joining is brittle, because two separately
  predicted outlines must fuse along a shared boundary. v0 fused
  shapes by compositing before tracing; direct outlines would need
  robust curve booleans, which the Rust ecosystem does not really
  have. Not recommended for v1.

### d. Hybrid: skeleton for structure, learned residual for finish

Predict the skeleton, expand the stroke, then apply a small learned
displacement to the expanded outline for terminals and optical
corrections. Probably the eventual answer (it is the img2bez
philosophy: procedural 90%, learned 9%, human 1%), but it depends on
(b) existing first.

### Recommendation

Two stages.

- **v0.5, the next demo: representation (a).** SDF plus img2bez
  tracing. Shortest path to naskh on screen, reuses the existing
  engine and the existing tracer, and the training data is free.
- **v1, the format-defining version: representation (b).** Skeleton +
  pen, hand-authored and editor-refined exemplars. The v0.5 model
  becomes the reference to proof against.

## 3. Training data

Three sources, in increasing order of ambition.

### a. Distill an existing OpenType naskh font

Shape a corpus with [rustybuzz](https://github.com/harfbuzz/rustybuzz)
(a complete pure-Rust HarfBuzz port, shaping-identical in practice),
render each glyph-in-context with ttf-parser outlines, and train on
(context → shape) pairs. The font's GSUB/GPOS machinery already
encodes the contextual rules; we would be distilling the best of
OpenType into a neural font, then generalizing past it.

- Source fonts, all OFL: [Amiri](https://github.com/aliftype/amiri)
  (Bulaq-press naskh, the richest contextual behavior), Scheherazade
  New (SIL, systematic and simpler), Noto Naskh Arabic.
- Corpus: a synthetic all-pairs wordlist (every letter before and
  after every letter) plus real text. A few thousand contexts.
- Licensing: a model trained to reproduce an OFL font is a derivative
  work. Release under OFL with a new name (Reserved Font Name rules
  apply: it cannot be called Amiri). Acceptable for research; state it
  plainly.
- This is days of engineering, not weeks, and it answers the central
  question early: **can a small network hold naskh at all, and at what
  parameter count?**

### b. Hand-authored skeletons on the nuqta grid

The v0 approach, upgraded from ASCII art to drawn centerlines. The
classical pedagogy is already a grid system: Ibn Muqla's proportioned
script (al-khatt al-mansub, 10th century) measures every letter in
rhombic dots of the pen (nuqat); alif is so many dots tall, the bowls
so many dots wide. A nuqta-quantized skeleton set is the traditional
analog of Virtua Grotesk's dyadic grid, and the same argument applies:
grid-disciplined sources are machine-legible training data.

- Authoring tool: Runebender-web already draws béziers; skeletons are
  open contours plus a width per point (storable in UFO lib data or a
  second layer). This is the beginning of the editor described in the
  blog post, where corrections in context become training samples.
- Effort: roughly 100 to 300 exemplars for a credible naskh subset,
  drawn by a person who can draw naskh. Slower than (a), but every
  sample is a design decision rather than an inherited one.

### c. Manuscript scans

The stated goal (the best naskh and nastaliq manuscripts on the web)
and a research project in itself: segmentation, skeleton extraction
from ink, normalization across hands. img2bez points in this
direction (it exists to trace rasters), but this is v2+ data. Not for
the next demo.

### Recommendation

(a) for v0.5, immediately. (b) begins in parallel as the editor work
matures, because v1's skeleton head needs it. (c) stays on the
horizon.

## 4. Joining, and the nastaliq cascade

The deep problem. Two mechanisms, one per representation.

### Compositing (for the SDF path)

OpenType naskh fonts already join the way v0 joins: each glyph's
baseline stroke ends in a flat edge at the advance boundary, and
adjacent glyphs abut. So per-glyph fields rendered from a shaped font
carry the same convention, and pointwise-max compositing fuses them
exactly as v0's bitmaps fused. Nothing new is needed for v0.5.

### Pen-state threading (for the skeleton path, and for nastaliq)

Generation flows through the word the way a calligrapher writes it.
Each glyph model takes (letter, neighbors, elongation, **incoming pen
state**) and produces (skeleton, **outgoing pen state**), where pen
state is position, tangent, and width at the connection. The engine
threads the state from glyph to glyph; joins are continuous by
construction, the way v0 made merging an invariant rather than a hope.

This also is the nastaliq answer. The cascade (each letter starting
lower than the last, the word sloping down to the baseline) is exactly
a pen state whose vertical component accumulates. Layout measures the
word's total descent in a first pass and starts the word high so it
lands on the baseline, which is how nastaliq actually behaves. For
nastaliq data, Gulzar
([googlefonts/Gulzar](https://github.com/googlefonts/Gulzar), OFL,
pure OpenType) and Noto Nastaliq Urdu both encode entry/exit anchors
as cursive attachment; rustybuzz reports the resulting offsets, so the
cascade geometry is extractable per glyph pair.

Nastaliq collisions (dots and bowls striking the next word's stack)
are the hardest known problem in this script; SIL's Awami Nastaliq
documentation is the honest reference. The demo answer is to ignore
collisions and say so.

## 5. Conditioning and architecture

v0 conditions on (letter, joining form, elongation): 63 inputs. Naskh
needs the neighbors, because the whole point is context beyond four
forms:

- letter id, joining form, elongation (as today)
- previous letter id and next letter id, as small learned embeddings
  (8 to 16 dimensions each) rather than one-hots
- later: position in word, style axes

Measured, not guessed: shape an all-pairs corpus through Amiri and
count distinct outlines per (letter, form). That number tells us how
much contextual variance the model must hold and becomes the v0.5
fidelity denominator, the naskh equivalent of 437/437.

Size budgets:

| Head | Output | Params (est.) | .ntf size |
| --- | --- | --- | --- |
| SDF 64×56 (dense) | 3,584 floats | ~1M | ~4 MB f32, ~1 MB int8 |
| SDF 64×56 (small deconv) | 3,584 floats | 100k to 300k | 0.4 to 1.2 MB |
| Skeleton + pen | ~100 floats | 50k to 120k | 200 to 500 KB |

The SDF head is demo-acceptable and webfont-marginal; the skeleton
head returns to real webfont budgets. Both stay trivially realtime on
CPU (a forward pass is on the order of a million multiply-adds).
Quantization (f16/int8 fields in the header) becomes worth doing at
v0.5 sizes.

## 6. Quality: what replaces 437/437

Continuous outputs need layered metrics:

1. **Field/geometry error** against held-out contexts: IoU of the
   thresholded field, or mean point error in font units for skeletons.
2. **Structural checks** on the traced outline, borrowed from
   img2bez's judge: points at extrema, no self-intersections, handle
   geometry sane, point-count parsimony.
3. **Proof sheets**, teacher vs model, per context, ranked worst
   first, exactly like the v0 sheets and the img2bez eval harness.
   Agents can grind this loop; a person judges the top of the ranking.
4. **Reading test**: running text at reading sizes next to the source
   font. The demo claim should be "indistinguishable at reading
   sizes," which is checkable by eye and honest about zoom.

## 7. The concrete plan

### v0.5: naskh distilled (the next demo)

1. **Extraction**: rustybuzz + ttf-parser over Amiri. Shape an
   all-pairs corpus plus real text; for every glyph-in-context record
   (letter, form, prev, next, outline, advance). Dedupe outlines to
   measure the true context count.
2. **Fields**: render each glyph outline to a 64×56 (try 96×84) SDF
   with kurbo/tiny-skia plus a distance transform, normalized to the
   same right-aligned canvas convention as v0.
3. **Model**: inputs from section 5, small deconv or dense head,
   trained with the existing hand-rolled trainer (add minibatching if
   full batch stops fitting).
4. **Engine**: GlyphImage grows a continuous variant; compositing
   becomes pointwise max; tracing goes through img2bez (or, first
   pass, marching squares + kurbo curve fit, then compare).
5. **Metrics**: section 6; the fidelity line becomes "N of M contexts
   within tolerance, worst context shown."
6. **Demo**: the same island with a font picker; the .ntf header
   already carries format/style, and this is the moment to add grid
   dimensions to the header so the two fonts can differ.
7. **License**: publish the distilled font as OFL with a new name.

### v1: skeleton + pen

1. Variable-width stroke expansion in kurbo (offset curves with
   varying distance; evaluate against Kalliculator-style output).
2. Pen-state threading in the engine, ports as a format contract.
3. Skeleton authoring in Runebender-web (open contours + widths in
   UFO); `ntf import` for UFO exemplars.
4. Author the first 50 naskh skeletons on a nuqta grid; train the
   skeleton head; proof against the v0.5 SDF model.
5. The editor loop from the blog post starts here: correct a letter
   in context, retrain, diff.

### v2: nastaliq

Pen-state threading plus cascade layout; data from Gulzar or Noto
Nastaliq Urdu cursive attachments; collisions acknowledged, not
solved.

## 8. Open problems, stated plainly

- **SDF resolution vs naskh detail.** 64px em may blur thin joins;
  96px triples the output size. Needs one experiment, early.
- **Trace cost per keystroke.** img2bez is milliseconds on big
  rasters; a composited line field is small. Likely fine, unmeasured.
- **Skeleton extraction from outlines** (if we ever want skeletons
  from Amiri rather than by hand): medial-axis computation is noisy
  exactly where it matters, at joins and terminals. This is why v1
  plans hand-authored skeletons instead.
- **How small can naskh get?** The central open question. If a
  50k-parameter skeleton model can hold Ottoman-quality naskh, the
  thesis of the whole project holds. If it takes 5M parameters, the
  format is still interesting but the story changes.
- **Terminal quality.** Pen models flatten terminals; hybrid residual
  (2d) is the planned answer, unproven here.
- **Cursor and clusters** carry over unchanged; the span machinery
  already handles ligatures, and nastaliq's cascade does not change
  the logical model.

## 9. Additions to the reading list

Beyond docs/RESEARCH.md: Ibn Muqla's proportional system (the nuqta
grid, via Sheila Blair's *Islamic Calligraphy*); Kalliculator
(Frederik Berlaen) as the closest prior art for pen-model font
generation; Raph Levien's parallel-curve and stroke-expansion writing
(kurbo's offset machinery); SIL's Awami Nastaliq engineering notes on
collision avoidance; [Gulzar](https://github.com/googlefonts/Gulzar)
and [Amiri](https://github.com/aliftype/amiri) as the OFL sources for
distillation.
