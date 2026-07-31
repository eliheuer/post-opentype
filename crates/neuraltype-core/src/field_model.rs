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
    cache: std::cell::RefCell<HashMap<[u32; 5], std::rc::Rc<GlyphField>>>,
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
        let vocab = header
            .vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        Ok(FieldFont {
            vocab,
            arch: header.arch,
            canvas: header.canvas,
            t,
            cache: Default::default(),
        })
    }

    pub fn vocab_id(&self, s: &str) -> Option<u32> {
        self.vocab.get(s).copied()
    }

    pub fn none_id(&self) -> u32 {
        0
    }

    /// One forward pass for one letter-in-context (cached).
    pub fn glyph(&self, feats: [u32; 5]) -> std::rc::Rc<GlyphField> {
        if let Some(g) = self.cache.borrow().get(&feats) {
            return g.clone();
        }
        let g = std::rc::Rc::new(self.forward(feats));
        self.cache.borrow_mut().insert(feats, g.clone());
        g
    }

    fn forward(&self, feats: [u32; 5]) -> GlyphField {
        let a = &self.arch;
        let emb = &self.t["emb.weight"];
        // concat embeddings
        let mut x = Vec::with_capacity(5 * a.emb);
        for &id in &feats {
            let base = id as usize * a.emb;
            x.extend_from_slice(&emb.data[base..base + a.emb]);
        }
        // l1 + relu
        let z = dense(&x, &self.t["l1.weight"], &self.t["l1.bias"], true);
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
