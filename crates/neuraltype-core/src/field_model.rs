//! Inference for "neuraltype-field-v1" fonts: hand-rolled, no
//! framework. The font file carries context embeddings, a small MLP,
//! and a deconvolution decoder; a forward pass produces a
//! signed-distance field for one letter-in-context plus the
//! displacement to the next letter (the cascade). Words compose by
//! chaining displacements and taking the pointwise max of fields.

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct TensorMeta {
    name: String,
    shape: Vec<usize>,
}

#[derive(Deserialize)]
struct Arch {
    emb: usize,
    latent: usize,
    c0: usize,
    grid0: [usize; 2],
    chans: Vec<usize>,
    kernel: usize,
    stride: usize,
    padding: usize,
    /// Conditioning inputs after the embeddings: 4 when the font was
    /// trained to draw pulled clusters (see `stretch`), else 0.
    #[serde(default)]
    cond: usize,
}

#[derive(Deserialize)]
pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub origin_x: f64,
    pub origin_y: f64,
    pub em_px: f64,
    pub upm: f64,
    pub spread_px: f64,
}

#[derive(Deserialize)]
struct Header {
    format: String,
    vocab: Vec<String>,
    arch: Arch,
    canvas: Canvas,
    tensors: Vec<TensorMeta>,
    /// Space contexts the font was trained on, as
    /// [prev2, prev, next, next2] ("" = none). A space in one of
    /// these contexts chains the next word to the previous one (a
    /// composed phrase); any other space is an ordinary word break.
    #[serde(default)]
    space_ctx: Vec<[String; 4]>,
}

struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

pub struct FieldFont {
    vocab: HashMap<String, u32>,
    arch: Arch,
    pub canvas: Canvas,
    t: HashMap<String, Tensor>,
    space_ctx: std::collections::HashSet<[u32; 4]>,
    cache: std::cell::RefCell<HashMap<[u32; 5], std::rc::Rc<GlyphField>>>,
    /// Pulled fields, keyed by context and the pull in whole pixels.
    pulled_cache: std::cell::RefCell<HashMap<([u32; 5], [i32; 4]), std::rc::Rc<GlyphField>>>,
}

pub struct GlyphField {
    /// SDF, row-major h×w, positive inside, in [-1, 1].
    pub field: Vec<f32>,
    /// Displacement to the next cluster origin (or, for a word's
    /// first cluster, its absolute origin), in font units.
    pub ddx: f64,
    pub ddy: f64,
}

pub fn is_field_font(bytes: &[u8]) -> bool {
    if bytes.len() < 8 || &bytes[..4] != b"NTF0" {
        return false;
    }
    let hlen = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    serde_json::from_slice::<serde_json::Value>(&bytes[8..8 + hlen])
        .ok()
        .and_then(|v| v["format"].as_str().map(|s| s.starts_with("neuraltype-field")))
        .unwrap_or(false)
}

impl FieldFont {
    pub fn load(bytes: &[u8]) -> Result<FieldFont, String> {
        if bytes.len() < 8 || &bytes[..4] != b"NTF0" {
            return Err("not a NeuralType file".into());
        }
        let hlen = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let header: Header =
            serde_json::from_slice(&bytes[8..8 + hlen]).map_err(|e| e.to_string())?;
        if header.format != "neuraltype-field-v1" {
            return Err(format!("unsupported format {}", header.format));
        }
        let mut off = 8 + hlen;
        let mut t = HashMap::new();
        for tm in &header.tensors {
            let n: usize = tm.shape.iter().product();
            let end = off + n * 4;
            if end > bytes.len() {
                return Err("truncated field font".into());
            }
            let data: Vec<f32> = bytes[off..end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            t.insert(tm.name.clone(), Tensor { shape: tm.shape.clone(), data });
            off = end;
        }
        let vocab: HashMap<String, u32> = header
            .vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        let space_ctx = header
            .space_ctx
            .iter()
            .map(|c| {
                let id = |s: &String| vocab.get(s).copied().unwrap_or(0);
                [id(&c[0]), id(&c[1]), id(&c[2]), id(&c[3])]
            })
            .collect();
        Ok(FieldFont {
            vocab,
            space_ctx,
            arch: header.arch,
            canvas: header.canvas,
            t,
            cache: Default::default(),
            pulled_cache: Default::default(),
        })
    }

    pub fn n_params(&self) -> usize {
        self.t.values().map(|t| t.data.len()).sum()
    }

    /// The letters this font supports (single chars from the vocab).
    pub fn alphabet(&self) -> String {
        let mut v: Vec<&String> = self.vocab.keys().collect();
        v.sort();
        v.iter().filter(|s| s.chars().count() == 1).cloned().cloned().collect()
    }

    pub fn vocab_id(&self, s: &str) -> Option<u32> {
        self.vocab.get(s).copied()
    }

    pub fn none_id(&self) -> u32 {
        0
    }

    /// Whether a space with these neighbors ([prev2, prev, next,
    /// next2] vocab ids) was trained as part of a composed phrase.
    pub fn space_trained(&self, ctx: [u32; 4]) -> bool {
        self.space_ctx.contains(&ctx)
    }

    /// One forward pass for one letter-in-context (cached).
    pub fn glyph(&self, feats: [u32; 5]) -> std::rc::Rc<GlyphField> {
        if let Some(g) = self.cache.borrow().get(&feats) {
            return g.clone();
        }
        let g = std::rc::Rc::new(self.forward(feats, [0.0; 4]));
        self.cache.borrow_mut().insert(feats, g.clone());
        g
    }

    /// Whether the network takes pulls as inputs (see `stretch`).
    pub fn learned_stretch(&self) -> bool {
        self.arch.cond == 4
    }

    /// One forward pass for a letter whose neighbors are pulled:
    /// `cond` is [prev x, prev y, next x, next y] in em, y down
    /// (cached per whole pixel). Without pulls this is `glyph`.
    pub fn glyph_pulled(&self, feats: [u32; 5], cond: [f32; 4]) -> std::rc::Rc<GlyphField> {
        if !self.learned_stretch() || cond == [0.0; 4] {
            return self.glyph(feats);
        }
        let em = self.canvas.em_px as f32;
        let key = (feats, cond.map(|c| (c * em).round() as i32));
        if let Some(g) = self.pulled_cache.borrow().get(&key) {
            return g.clone();
        }
        let g = std::rc::Rc::new(self.forward(feats, cond));
        let mut cache = self.pulled_cache.borrow_mut();
        // a drag visits many pulls; keep the cache from growing forever
        if cache.len() > 512 {
            cache.clear();
        }
        cache.insert(key, g.clone());
        g
    }

    fn forward(&self, feats: [u32; 5], cond: [f32; 4]) -> GlyphField {
        let a = &self.arch;
        let emb = &self.t["emb.weight"];
        // concat embeddings
        let mut x = Vec::with_capacity(5 * a.emb + a.cond);
        for &id in &feats {
            let base = id as usize * a.emb;
            x.extend_from_slice(&emb.data[base..base + a.emb]);
        }
        x.extend_from_slice(&cond[..a.cond.min(4)]);
        // l1 + relu
        let mut z = dense(&x, &self.t["l1.weight"], &self.t["l1.bias"], true);
        // optional second hidden layer
        if let (Some(w), Some(b)) = (self.t.get("l1b.weight"), self.t.get("l1b.bias")) {
            z = dense(&z, w, b, true);
        }
        // displacement head (font units)
        let d = dense(&z, &self.t["disp.weight"], &self.t["disp.bias"], false);
        let (ddx, ddy) = (d[0] as f64 * self.canvas.upm, d[1] as f64 * self.canvas.upm);
        // l2 + relu, reshape to (c0, g0h, g0w)
        let mut cur = dense(&z, &self.t["l2.weight"], &self.t["l2.bias"], true);
        let (mut ch, mut gh, mut gw) = (a.c0, a.grid0[0], a.grid0[1]);
        // deconv chain
        for i in 0..a.chans.len() - 1 {
            let w = &self.t[&format!("d{i}.weight")];
            let b = &self.t[&format!("d{i}.bias")];
            let out_ch = a.chans[i + 1];
            let oh = gh * a.stride;
            let ow = gw * a.stride;
            let mut out = vec![0.0f32; out_ch * oh * ow];
            // bias
            for oc in 0..out_ch {
                let v = b.data[oc];
                out[oc * oh * ow..(oc + 1) * oh * ow].fill(v);
            }
            // weight shape: (in_ch, out_ch, k, k)
            let k = a.kernel;
            for ic in 0..ch {
                let in_plane = &cur[ic * gh * gw..(ic + 1) * gh * gw];
                for oc in 0..out_ch {
                    let wbase = ((ic * out_ch) + oc) * k * k;
                    let wk = &w.data[wbase..wbase + k * k];
                    for iy in 0..gh {
                        let oy0 = iy * a.stride;
                        for ix in 0..gw {
                            let v = in_plane[iy * gw + ix];
                            if v == 0.0 {
                                continue; // relu sparsity
                            }
                            let ox0 = ix * a.stride;
                            for ky in 0..k {
                                let oy = oy0 + ky;
                                if oy < a.padding || oy - a.padding >= oh {
                                    continue;
                                }
                                let orow = (oc * oh + (oy - a.padding)) * ow;
                                let wrow = ky * k;
                                for kx in 0..k {
                                    let ox = ox0 + kx;
                                    if ox < a.padding || ox - a.padding >= ow {
                                        continue;
                                    }
                                    out[orow + ox - a.padding] += v * wk[wrow + kx];
                                }
                            }
                        }
                    }
                }
            }
            let last = i + 1 == a.chans.len() - 1;
            if !last {
                out.iter_mut().for_each(|v| {
                    if *v < 0.0 {
                        *v = 0.0
                    }
                });
            }
            cur = out;
            ch = out_ch;
            gh = oh;
            gw = ow;
        }
        // crop (gh,gw) -> canvas (h,w)
        let (h, w) = (self.canvas.h, self.canvas.w);
        let mut field = vec![0.0f32; h * w];
        for y in 0..h {
            field[y * w..(y + 1) * w].copy_from_slice(&cur[y * gw..y * gw + w]);
        }
        GlyphField { field, ddx, ddy }
    }
}

fn dense(x: &[f32], w: &Tensor, b: &Tensor, relu: bool) -> Vec<f32> {
    // candle Linear: weight shape (out, in), y = W x + b
    let out_dim = w.shape[0];
    let in_dim = w.shape[1];
    assert_eq!(x.len(), in_dim);
    let mut y = b.data.clone();
    for o in 0..out_dim {
        let row = &w.data[o * in_dim..(o + 1) * in_dim];
        let mut acc = 0.0f32;
        for (wi, xi) in row.iter().zip(x) {
            acc += wi * xi;
        }
        y[o] += acc;
        if relu && y[o] < 0.0 {
            y[o] = 0.0;
        }
    }
    y
}
