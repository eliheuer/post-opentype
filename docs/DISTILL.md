# Naskh Distilled: Converting TTFs to NTFs

**Status: plan for stage 0** (see [NASKH.md](NASKH.md) section 7).
Convert Amiri, an OFL naskh with the richest contextual behavior in
open fonts, into a .ntf neural font. The output is a demo, a blog
post ("Naskh Distilled: converting TTFs to NTFs"), and every pipeline
component the manuscript work needs, proven on free data.

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

Order of targets: Amiri first (a decade of production hardening makes
it the safe pipeline target), Gulzar second (nastaliq, where the
cascade begins).

## Open questions to settle by experiment

- Field resolution vs naskh hairlines (64 vs 96 em pixels).
- Dense vs deconv head; parameter count vs fidelity curve.
- Whether prev/next letter embeddings capture Amiri's contextual
  substitutions, measured against the deduped variance count from
  step 1.
- Trace cost per keystroke at line width (budget: under a frame).
- Quantization: f16 and int8 field heads vs fidelity.
