# Nastaliq Distilled: Converting TTFs to NTFs

**Status: plan for stage 0** (see [NASKH.md](NASKH.md) section 7).
Convert Gulzar, the OFL nastaliq, into a .ntf neural font. Nastaliq
is the style OpenType limits most, so it is the demo that matters.
The output is a demo, a blog post ("Nastaliq Distilled: converting
TTFs to NTFs"), and every pipeline component the manuscript work
needs, proven on free data. Amiri (naskh, calmer geometry) follows on
the same pipeline.

## Why distill, given that reproduction is not the goal

Reproducing an OpenType font proves nothing about the tradition. What
it proves is the pipeline. Stage 0 forces into existence, on data
that costs nothing:

- the extraction tooling (shaping a corpus, harvesting shapes in
  context),
- the continuous field output head (v0's grid, made grayscale and
  fine),
- field compositing and tracing in the engine,
- tolerance-based fidelity metrics (the continuous 437/437),
- quantization (the field head is too big for f32 to stay polite).

It also answers the sizing question early: how many parameters does
contextual naskh need? That number shapes everything after.

## Pipeline

1. **Extract.** [rustybuzz](https://github.com/harfbuzz/rustybuzz)
   (pure-Rust HarfBuzz port) + ttf-parser over
   [Amiri](https://github.com/aliftype/amiri). Shape an all-pairs
   corpus (every supported letter adjacent to every other, in every
   joining position) plus real text. For each glyph-in-context,
   record: letter, joining form, previous and next letter, glyph id,
   outline, advance. Dedupe outlines per (letter, form) to measure
   Amiri's true contextual variance; that count is the training
   denominator.
2. **Fields.** Render each glyph outline to a normalized distance
   field (64×56 first, 96×84 if detail demands it), right-aligned on
   the canvas per v0's convention. OpenType naskh joins by abutting
   flat stroke ends at the advance boundary, so v0's compositing
   convention carries over.
3. **Train.** Inputs: letter id, joining form, prev/next letter
   embeddings, elongation (tatweel contexts give real elongation
   samples). Head: dense first, small deconv if parameters balloon.
   The existing hand-rolled trainer grows minibatching.
4. **Engine.** GlyphImage gains a continuous field variant;
   compositing becomes pointwise max; the traced outline comes from
   the composited line, supersampled, through img2bez (fallback:
   marching squares + kurbo fitting, compared side by side).
5. **Metrics.** Held-out IoU per context, worst-context proof sheets
   ranked first, img2bez-style structure checks on traced outlines,
   and a reading proof against Amiri itself. Fidelity line: "N of M
   contexts within tolerance."
6. **Demo.** The existing island with a font picker (kufic / naskh).
   Prerequisite: grid dimensions move into the .ntf header so fonts
   can differ (also fixes the header's honesty; see the format
   section of the blog post).
7. **License.** The distilled font is a derivative of Amiri: released
   OFL under a new name (Amiri is a Reserved Font Name).

## First measurement (2026-07-30)

The probe (`cargo run -p neuraltype-cli --bin probe`) shapes all
ordered letter pairs through Amiri with rustybuzz and counts distinct
glyphs per letter per position. Results: 150 distinct first-position
variants and 107 second-position variants across 31 letters, from
pairs alone. ب ت ن ي each take 10 different first-position glyphs
depending on the following letter; ر takes 8 final variants depending
on the preceding one. Medial contexts (to be measured with carrier
templates) will add more. Amiri also ships 6,710 glyphs total.

Gulzar, same probe: 344 distinct first-position variants (ن takes 16
initial forms, ي 15, م 14), more than double Amiri's contextual
variance. Cascade measured directly: shaping نستعليق places glyphs at
y-offsets up to 1.38 em above the baseline; even بسم spans 0.80 em.
Extraction implication: rustybuzz reports the cursive-attachment
offsets in glyph positions, so per-glyph training targets are (field,
displacement), and the engine chains displacements to produce the
cascade.

Order of targets: Gulzar first (nastaliq is the thesis case), Amiri
second (naskh regression test for the same pipeline).

## Tools

The extraction pipeline lives in `crates/neuraltype-distill`
(shaping via [harfrust](https://github.com/harfbuzz/harfrust), the
HarfBuzz-org Rust port at HarfBuzz 13 parity; outlines via
ttf-parser):

```sh
# shape the corpus through a font and write the dataset
cargo run --release -p neuraltype-distill --bin distill --     extract data/Gulzar-Regular.ttf data/extract-gulzar

# summarize: shape counts per letter, displacement ranges
cargo run --release -p neuraltype-distill --bin distill --     stats data/extract-gulzar
```

The corpus is every single letter, ordered pair, and ordered triple
over 31 Arabic letters (30,783 words); triples capture medial forms in
their full (previous, next) context. Output is JSONL, inspectable with
jq:

- `meta.json`: font, units per em, corpus and dedup counts.
- `glyphs.jsonl`: `{gid, path}`, one SVG path per glyph the corpus
  touched, font units, y-up.
- `contexts.jsonl`: one record per cluster occurrence: the word, the
  cluster's letters, logical index, prev/next letters, its glyphs
  (base plus marks) placed relative to the cluster origin, and the
  displacement `(ddx, ddy)` from the previous cluster's origin. The
  displacement chain is the cascade.

## Extraction results: Gulzar (2026-07-30)

- 30,783 words shaped → 91,263 cluster records.
- 411 unique glyphs touched; **1,223 unique composed cluster shapes**
  (base + dots), the training denominator.
- Per letter, composed: خ 108 distinct shapes, ث 100, ت 98, ي 80,
  ن 78. Dots multiply the base-form counts.
- Cluster displacement dy spans −1,662 to +1,190 font units at
  1000 upm: the cascade drops more than 1.6 em between adjacent
  letters at the extreme.
- rustybuzz (HarfBuzz 10) and harfrust (HarfBuzz 13) produce identical
  counts on this corpus, a useful cross-validation; the pipeline uses
  harfrust.

## Fields stage (built, 2026-07-30)

`distill fields <extract-dir> <out-dir> [em_px]` dedupes the context
records to unique composed shapes, rasterizes each (kurbo flattening,
nonzero scanline fill, 4× supersample), computes an exact Euclidean
signed-distance field (Felzenszwalb transform), clamps at 1/8 em, and
writes one u8 grid per shape plus `dataset.jsonl` mapping every
context row to its shape id. `distill proof <fields-dir> <out.pgm>
[ids…]` renders a proof sheet (convert with magick; threshold at 50%
to see the traced contour).

Gulzar results:

- 1,145 unique shapes (shape-level dedup; the 1,223 earlier counted
  letter-and-shape pairs, and some shapes are shared, e.g. ي/ى).
- Shape bbox relative to the cluster origin spans x −740..1427,
  y −1601..1570 font units: nastaliq swashes and cascade-carried marks
  make the cluster canvas large.
- Canvas at 64 px/em: **155×219 px**; dataset 38.9 MB u8.
- **Resolution decision: 64 px/em.** Thresholded contours at 64 hold
  Gulzar's tapers and hairlines; 96 is visually indistinguishable at
  reading scale and costs 2.3× the outputs (a 96 dataset is also
  generated for a fidelity ablation later).
- Implication for the model head: 155×219 = 33,945 outputs per shape.
  A dense final layer at that width is ~10M parameters, so the train
  stage starts with a small deconv decoder, and a cropped or
  origin-tightened canvas is the fallback.

## Comparison images (planned)

Quality and file-size comparisons for the blog post, generated with
designbot in the style of the Virtua Grotesk specimen proofs (dark
ground, subtle grid, light letterforms): Gulzar-via-OpenType vs
Gulzar-via-.ntf renders of the same text side by side, with a
file-size table (Gulzar-Regular.ttf is 963 KB; the .ntf size is a
result we do not have yet).

## Open questions to settle by experiment

- Field resolution vs naskh hairlines (64 vs 96 em pixels).
- Dense vs deconv head; parameter count vs fidelity curve.
- Whether prev/next letter embeddings capture Amiri's contextual
  substitutions, measured against the deduped variance count from
  step 1.
- Trace cost per keystroke at line width (budget: under a frame).
- Quantization: f16 and int8 field heads vs fidelity.
