# NeuralType (.ntf)

The NeuralType font format (.ntf) is an experiment in what comes after
OpenType: a font format where the font file is a small neural network
that draws each letterform in context, instead of a table of fixed
outlines to look up.

Two posts cover the work, both with live demos you can type in:

- [NeuralType: A Post-OpenType Font Format](https://elih.net/blog/neuraltype/)
  introduces the format and the engine, using a square Kufic font that
  is 53,600 weights in a 215 KB file.
- [Nastaliq Distilled](https://elih.net/blog/nastaliq-distilled/) is the
  build log for the harder case: converting Gulzar, an OFL nastaliq,
  into a 1.36M-parameter model that draws each letterform and its place
  in the cascade from context. It also records the experiments that
  failed, including three attempts to improve the model by making it
  bigger.

Documentation lives in [docs/](docs/): `SPEC.md` for the format,
`DISTILL.md` for the distillation method, `HANDOFF.md` for the current
state of the work and how to pick it up, `VECTOR.md` for the
vector-output experiment, and `COMPRESSION.md` for the file-size work.

Dual-licensed under Apache-2.0 or MIT.
