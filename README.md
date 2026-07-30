# NeuralType (.ntf) — post-OpenType generative fonts

**A font that is a model, not a table of outlines.**

An OpenType font is, at heart, a compression scheme for the Latin
letter: one fixed outline per glyph, reused everywhere, boxed into
per-glyph cells inherited from metal type. Scripts that were never
made of boxes strain against this. Arabic in its manuscript forms —
naskh, and above all nastaliq — is one continuous, context-dependent
stroke; the OpenType workaround is thousands of glyphs and
substitution rules faking what a hand does in one motion.

NeuralType is the alternative: the font file is the weights of a small
neural model that draws each glyph *in context*, on the fly. No glyph
tables — rendering is a forward pass followed by outline extraction.
The format is script-agnostic (TrueType → OpenType → NeuralType); the
Arabic script is the first target because it is where per-glyph
outlines fail hardest, and square Kufic is the first style because its
grid discipline makes every part of the pipeline inspectable.

## What works today (v0)

A complete end-to-end demo for **square Kufic** (the grid-based style
is deliberately the first target — its letterforms live on a coarse
binary grid, which makes the generative representation trivial and the
concept legible):

- `crates/neuraltype-core` — the engine (Rust, linebender/kurbo):
  - a procedural **teacher**: square-Kufic letterforms authored as
    ASCII-art grids, composed with joining behaviour, i'jam dots, and
    continuous kashida elongation;
  - a **neural font**: a 53,600-parameter MLP mapping
    *(letter, joining form, elongation) → occupancy grid*; the advance
    width derives from the generated grid;
  - Arabic **shaping** (joining-form analysis) and RTL layout;
  - a **tracer** that converts generated grids into vector outlines.
- `crates/neuraltype-cli` — `ntf train` distills the teacher into a
  `.ntf` font file (weights only, ~200 KB f32); `ntf sheet` /
  `ntf render` produce SVG proofs comparing teacher and model.
- `crates/neuraltype-wasm` + `demo/` — a browser demo: type Arabic text,
  every outline is generated live by the model in WASM, with a
  continuous elongation slider (a parameter OpenType cannot express).

```sh
cargo run --release -p neuraltype-cli --bin ntf -- train
wasm-pack build crates/neuraltype-wasm --target web --out-dir ../../demo/pkg --release
cp build/kufic.ntf demo/
python3 -m http.server -d demo 8123   # open http://localhost:8123
```

Current fidelity: 437/437 (letter, form, elongation) contexts —
Arabic including the لا ligature and hamza, plus Latin capitals —
reproduced exactly by the model.

## Where this goes

The ladder (see [docs/SPEC.md](docs/SPEC.md)):

1. **v0 — square Kufic, occupancy grid** (this repo): prove the
   pipeline — text → context → model → outlines.
2. **v1 — naskh, bezier control points**: same conditioning, output
   becomes skeleton control points + pen widths; stroke-expanded with
   kurbo. Context window widens to neighbouring letters so forms adapt
   beyond the four OpenType classes.
3. **v2 — nastaliq**: sloped baselines, vertical stacking, contextual
   kerning — the script OpenType handles worst, and the one a
   contextual generative model is naturally shaped for.

Research notes and prior art: [docs/RESEARCH.md](docs/RESEARCH.md).
