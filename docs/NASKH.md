# v1 Research: Manuscript Nastaliq on the Web

**Status: research, revision 2.** The first revision of this document
proposed distilling an existing OpenType naskh font and building a
skeleton-plus-pen output head. Both were rejected on review; the
record of why is in section 8. This revision starts from the actual
goal.

## 1. The goal

Not to reproduce existing nastaliq fonts. Existing digital nastaliq is
already a compromise: the script forced into the emulated metal-type
sorts of OpenType. Reproducing it with a neural network would
demonstrate nothing.

The goal is the first digital fonts that do the nastaliq tradition
justice: manuscript-quality words on the web that are still real text.
Selectable, clickable, copyable, pastable, exactly as the v0 demo
already does for square Kufic. And beyond OpenType's reach: words that
can be moved, scaled, and refit, with the letterforms regenerating to
suit their context instead of being squeezed or stretched as frozen
outlines.

The reference standard is the manuscript tradition itself: muraqqa and
qit'a panels where words nest, stack, and fill their frame, and where
the quality lives in the balance of black and white.

## 2. Design principle: notan

Good typography is figure and ground, black and white, notan. The
tradition's quality judgment operates on the masses, not on an
imagined centerline. This is the representation criterion for the
whole project:

**The model must learn and output in the shape domain, where figure
and ground are symmetric. Not in the stroke domain.**

The practical consequence is that the field representation from v0 is
the right lineage, not a stepping stone. A signed-distance field is
notan as data: the sign is figure versus ground, the zero contour is
the boundary, and nothing about it assumes the shape came from a pen.
v0's binary occupancy grid was the coarsest possible notan
representation; v1 refines it instead of replacing it.

The pen is not discarded; it is demoted to what it really is: a prior.
The reed pen explains why the masses look the way they do, and that
knowledge is useful for data augmentation and for editor guides. It is
never the output representation, because the finished manuscript
letterform is not a pen sweep; it is a judged and corrected shape.

## 3. The unit of generation: the word

v0 generates per glyph and composites on a shared baseline row.
Nastaliq words do not decompose that way; a word is one cascading
gesture whose geometry depends on everything in it. So the natural
unit of generation is the **word**:

```
word_shape = f(letters of the word, fit parameters, style)
```

This dissolves the hardest v0-style problem. Intra-word joining stops
being an engineering contract (ports, welds, shared cells) and becomes
part of the learned shape, which is what it is in the manuscripts.

What word-level generation must not break is the text machinery, and
this needs one new idea. In v0 the engine reports per-character spans
for cursor, selection, and clipboard. A generated word-shape needs the
2D equivalent: alongside the shape field, the model outputs a **letter
map**, a soft segmentation channel that assigns regions of the word to
logical letter indices. Cursor placement, hit testing, and selection
highlighting read the letter map; the hidden-input architecture from
v0 carries over unchanged. Copy and paste never see shapes at all.

Word boundaries remain the engine's job (spaces are layout, not
generation), so lines compose from generated words the way v0 lines
compose from glyphs.

## 4. Representation

- **Output: a field.** Signed-distance (or occupancy at higher
  resolution) on a normalized canvas per word, plus the letter map
  channel, plus anchor metadata (baseline entry and exit, so layout
  can place the word). Resolution is an experiment: 128 to 256 pixels
  of em height is the plausible range for manuscript detail.
- **Vector output stays.** The composited field is traced to bézier
  outlines at render time. This is img2bez's problem statement
  verbatim: raster in, font-quality outline out, deterministic, in
  milliseconds. The tracing stage is the one part of this project that
  is already built.
- **Conditioning.** Letter sequence of the word (variable length, so
  the head grows past a flat MLP: a small sequence encoder feeding the
  field decoder), fit parameters (width target, the generalization of
  v0's elongation), and eventually style axes.
- **Determinism holds.** No sampling. Same word, same parameters, same
  field, everywhere. The v0 claim survives intact.

Honest size estimate: this is no longer a 53k-parameter MLP. A
text-conditioned field decoder for one style is plausibly one to ten
million parameters, a few megabytes quantized. Chunky for a webfont,
viable for a demo, and the number is itself a research result: how
small can a hand be?

## 5. Data: manuscripts first

The first revision deferred manuscripts as too hard. Reviewing the
goal, they are not optional: the manuscript tradition is the standard,
so it must be the data.

1. **Corpus.** Digitized nastaliq manuscripts and panels. Museum and
   library collections (public-domain digitizations) plus the
   designer's own collection of reference scans. Start embarrassingly
   small: a few panels is a few hundred words.
2. **Segmentation.** Binarize ink from ground, segment words. The
   panels are high-contrast by design; classical vision goes far
   before any learning is needed.
3. **Transcription.** Each word labeled with its letter sequence. At
   corpus sizes of hundreds to low thousands of words this is
   designer-scale labor, not crowd-scale.
4. **Normalization.** Scale to a common em, orient to the cascade,
   compute fields. Letter maps for training come from coarse manual
   segmentation on a subset plus propagation, and improve through the
   editor loop.
5. **Augmentation.** The pen prior earns its keep here: synthetic
   variations (slight rotations, width modulation consistent with a
   nib) multiply a small corpus without leaving the style.

The second data stream is the editor (section 6): every correction
and every newly drawn exemplar is a labeled training sample. The
corpus is not collected once; it accumulates as a byproduct of design
work. Grid-disciplined, machine-legible sources as the design medium
is the Virtua Grotesk thesis, applied to a millennium-old style.

## 6. The editor: masses, not points

No existing font editor can author this format, which is the point.
The editor for a notan-domain font edits masses:

- The canvas is the field itself. Corrections are painted and carved
  (figure and ground are both first-class), not dragged point by
  point. The vector outline is a live traced preview, never the
  source.
- Editing happens in context: type a word, the model draws it, the
  designer corrects the word, and the correction is stored with its
  full conditioning as a training sample.
- Active learning as proofing: the editor surfaces the words the model
  is least certain of, ranked, the way the img2bez eval harness ranks
  worst-first.
- Train is the compile step, minutes not hours, with proof sheets
  against the exemplar corpus.

Runebender-web is the natural host (canvas, text plumbing, live
reload already exist), with the mass-editing surface as the new part.
This editor does not exist anywhere; building it is the frontier
claim of the project.

## 7. Staging

- **Stage 0: nastaliq distilled (the next demo and blog post).**
  Convert Gulzar, the OFL nastaliq, to .ntf: shape an all-pairs corpus
  with rustybuzz, render per-glyph fields plus cursive-attachment
  displacements, train the field head, composite and trace in the
  engine, publish under a new OFL name. Amiri (naskh) follows on the
  same pipeline. This is not the goal (section 8 records why reproduction
  proves nothing about the tradition); it is the pipeline rehearsal.
  Every component it forces into existence, the extraction tooling,
  the continuous field head, the field compositing, the img2bez
  tracing stage, the tolerance metrics, is exactly the component the
  manuscript work needs, exercised on data that is free. Plan in
  [docs/DISTILL.md](DISTILL.md). Working title for the write-up:
  "Nastaliq Distilled: converting TTFs to NTFs."
- **Stage 1: one hand, linear text.** Train the word-field model on a
  small transcribed corpus from one manuscript hand. Demo: type,
  select, copy, paste manuscript-quality nastaliq words in the
  existing island, with a fit slider showing regeneration under
  pressure. This alone is past what OpenType can do, and past what
  the distillation plan would have shown.
- **Stage 2: the editor loop.** Mass-domain correction in
  Runebender-web; the corpus grows; quality climbs measurably
  (held-out field error, traced-outline structure checks, reading
  proofs against the source panels).
- **Stage 3: composition.** Panel layout in the manuscript manner:
  words placed, scaled, and refit under a layout optimizer that can
  re-generate any word at any fit. Collisions become a soft cost in
  the optimizer rather than an unsolved font-engineering problem.

Naskh remains available as a lower-risk rehearsal of the same
pipeline (calmer cascade, simpler composition), but the target that
justifies the work is nastaliq.

## 8. Considered and rejected

Kept as a record, because both were serious candidates.

- **Skeleton + pen output head.** Rejected as a representation:
  stroke expansion privileges the centerline and makes the white
  space a residual, and the finished manuscript letterform is a
  corrected shape, not a pen sweep. Good design thinks in figure and
  ground. The pen survives as a data prior and as editor guides only.
  This also deletes the variable-width stroke-expansion engineering
  the first revision spent its risk budget on.
- **Distilling an existing OpenType naskh or nastaliq font, as the
  goal.** Rejected as a goal, adopted as a rehearsal (stage 0 above):
  existing digital nastaliq is the compromise the project exists to
  escape, and reproducing it proves nothing about the tradition. What
  it does prove is the pipeline, on free data, before the manuscript
  corpus exists.

## 9. Open problems, stated plainly

- **Transcription is the bottleneck.** The corpus starts at hundreds
  of words and grows by hand. The editor loop is the only scaling
  mechanism, and it has to be pleasant enough to use daily.
- **Field resolution versus manuscript detail.** Hairline terminals
  and tight counters at 128px em need measurement, early.
- **The letter map is unproven.** Soft 2D segmentation for cursor and
  selection inside a generated word is a new mechanism; it needs a
  prototype before anything depends on it.
- **Model size.** One to ten million parameters is an estimate with
  wide error bars in both directions. The answer is a headline result
  either way.
- **Style coherence across words.** Per-word generation must not
  drift: two words on one line must read as one hand. Conditioning on
  shared style state, or a consistency loss, are candidate answers.
- **What the demo owes the tradition.** Reproducing manuscript words
  a model was trained near is not yet doing the tradition justice;
  the bar is new words, unseen combinations, that a calligrapher
  would accept. That judgment cannot be automated and should not be.
