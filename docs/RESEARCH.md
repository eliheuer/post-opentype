# Research collection — fonts as models

Prior art and reference material for a post-OpenType generative font
format, with a focus on the Arabic script. Organized by thread; each
entry notes what it contributes to this project.

## 1. The critique: what per-glyph outlines cost Arabic

- **Thomas Milo — DecoType, ACE (Arabic Calligraphic Engine), and
  Tasmeem** (for Adobe InDesign ME). The strongest existing statement
  of this project's premise: Milo's analysis distinguishes the
  *rasm* (skeleton) from dots/vowels, and renders naskh from a
  script grammar rather than a glyph inventory. ACE composes letter
  fragments dynamically — hundreds of contextual variants generated
  from a compact description. Search: "Thomas Milo Arabic script
  grammar", "Tasmeem", DecoType's Unicode conference papers (e.g.
  "Arabic script and typography: a brief historical overview").
- **Nastaliq in OpenType** is the stress test: Monotype's Noto
  Nastaliq Urdu (~1,400 glyphs + heavy GPOS cursive attachment) and
  the SIL Awami Nastaliq write-ups (Graphite-based because OpenType
  wasn't enough) document the machinery required. Awami's
  documentation is candid about collision avoidance, kerning across
  the cascade, and why flat substitution rules fight the script.
- **The tatweel/kashida problem**: justification in Arabic is
  elongation, a continuous typographic parameter; OpenType offers a
  spacer glyph (U+0640) and JSTF tables almost nothing implements.
  See Titus Nemeth, *Arabic Type-Making in the Machine Age* (Brill,
  2017) — the history of how hot-metal and photocomposition
  constraints shaped (deformed) printed Arabic.

## 2. Parametric ancestors: letters as programs

- **Donald Knuth, Metafont / Computer Modern** (1979–84). Letters as
  pen strokes over parametrized skeletons; a whole family from ~60
  parameters. The direct intellectual ancestor: our model output *is*
  a skeleton + pen, with the parameter logic learned rather than
  hand-programmed. Knuth's "The Concept of a Meta-Font" (Visible
  Language, 1982) is the essay to cite.
- **AlQalam** (Ameer Sherif & Hossam Fahmy, TUGboat): Metafont for
  Arabic/naskh, explicitly aiming at Qur'anic quality; demonstrates
  both the power and the authoring cost of hand-parametrization —
  the cost a learned model amortizes.
- **Omar Aziz's "Nastaleeq: a challenge accepted"** and the Pakistani
  InPage/Noori Nastaliq lineage (ligature-per-word approach: tens of
  thousands of stored ligatures) — the reductio ad absurdum of the
  table approach.
- **Apple TrueType GX / OpenType 1.8 variable fonts** (2016):
  interpolation between outline masters. Continuous, but only along
  pre-authored axes with fixed point correspondence — a linear,
  hand-built version of what a generative model does nonlinearly.
- **Grid-based and modular Arabic type**: square Kufic itself is the
  historical proof that Arabic admits a grid discipline (architectural
  banna'i brickwork); modern modular experiments (e.g. by Mourad
  Boutros, Mamoun Sakkal's square Kufic studies — sakkal.com) are the
  design-space v0 lives in.

## 3. Learned glyph generation

- **FontRNN** (Tang et al., Computer Graphics Forum 2019) — RNN
  generating Chinese glyph stroke sequences.
- **DeepVecFont / DeepVecFont-v2** (Wang et al., SIGGRAPH Asia
  2021/2023) — dual-modality (raster + vector) generation of complete
  fonts; state of the art for learned bezier outline synthesis.
- **Im2Vec** (Reddy et al., CVPR 2021) — differentiable rasterization
  of bezier paths from latent codes, no vector supervision needed.
- **SVG-VAE** (Lopes et al., ICCV 2019), **DeepSVG** (Carlier et al.,
  NeurIPS 2020) — transformer/VAE architectures over SVG command
  sequences; DeepSVG's hierarchical representation is a candidate for
  the v1/v2 output head.
- **Attribute2Font** (Wang et al., SIGGRAPH 2020) — synthesis
  conditioned on style attributes: evidence for learned style axes.
- **Arabic-specific**: work on Arabic calligraphy synthesis is
  thinner — mostly raster GAN work on calligraphy images (e.g.
  "CalliGAN"-style papers, Arabic handwriting synthesis for OCR
  augmentation). The vector, shaping-integrated version is open
  territory — which is the opportunity.
- **Key difference from all of the above**: they generate *fonts*
  (per-glyph, context-free, offline). We generate *text setting* —
  the model runs inside the shaping loop, conditioned on context, at
  keystroke time. Closest in spirit is not the font-ML literature but
  DecoType's ACE.

## 4. The stack we build on

- **linebender**: `kurbo` (curves, stroke expansion — Nehab's 2020
  stroke-to-fill work landed as `kurbo::stroke`), `peniko`, `vello`
  (GPU 2D rendering; the natural v2 renderer), `parley`/`swash`
  (text layout — what a `.ntf` engine would eventually slot into).
- **HarfBuzz** `hb-shape` as the reference for Arabic joining-form
  logic (USE/arabic shaper); Unicode UAX #9 (bidi) and the Unicode
  Arabic block's joining classes (ArabicShaping.txt) — v0 implements
  a subset of exactly this.
- **WASM**: `wasm-bindgen`/`wasm-pack`; the whole engine + font must
  stay in webfont-budget territory (v0: 130 KB wasm + 200 KB font).

## 5. Design references for the styles

- Square Kufic: Mamoun Sakkal's analyses; banna'i architectural
  epigraphy (Samarkand, Isfahan); the classic pixel-chart tables of
  the four contextual forms (the direct model for v0's teacher art).
- Naskh: Ottoman mushaf hands (Hâfız Osman); DecoType Naskh as the
  digital benchmark.
- Nastaliq: Mir Emad's Persian canon; Noori Nastaliq and Awami
  Nastaliq as digital baselines; the sloped-baseline "cascade" as the
  core geometric fact a generative model must own.

## 6. Open research questions logged here

- Output representation for v1: fixed-slot skeleton points (simple,
  differentiable) vs. SVG-command sequence decoding (DeepSVG-style,
  variable length) vs. implicit fields + tracing (resolution-free but
  loses stroke semantics).
- Teacher for naskh: parametric engine (Metafont/AlQalam-style) vs.
  fitting manuscript scans (Im2Vec-style differentiable raster loss)
  vs. distilling an existing high-quality font's instantiated
  contexts (legal questions: outlines are copyrightable as programs
  in some jurisdictions; trained-weight status is untested).
- Whole-word vs. per-glyph generation for nastaliq: per-glyph with
  rich context keeps cursor/selection semantics; whole-word is truer
  to the script. Middle path: per-glyph outputs + a word-level
  "conductor" vector.
- Evaluation: cell/curve fidelity to teacher is v0's metric; real
  metric is legibility + naskh/nastaliq acceptability judged by
  readers and calligraphers.
