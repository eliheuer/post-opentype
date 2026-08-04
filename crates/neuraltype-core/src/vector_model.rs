//! Inference for "neuraltype-vector-v1" fonts: a decoder-only
//! transformer that emits quantized outline tokens (command + coord
//! deltas) for one letter-in-context, hand-rolled with a KV cache --
//! no framework. Placement comes from a per-context displacement
//! table in the file (the cascade), chained exactly like the field
//! format; a word-first context's entry holds its absolute origin.

use kurbo::BezPath;
use serde::Deserialize;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Deserialize)]
struct TensorMeta {
    name: String,
    shape: Vec<usize>,
}

#[derive(Deserialize)]
struct Arch {
    d: usize,
    layers: usize,
    heads: usize,
    ffn: usize,
    prefix: usize,
    max_pos: usize,
}

#[derive(Deserialize)]
struct Units {
    upm: f64,
    em_px: f64,
    q_units: f64,
}

#[derive(Deserialize)]
struct Header {
    format: String,
    tok_vocab: Vec<String>,
    letter_vocab: Vec<String>,
    arch: Arch,
    units: Units,
    tensors: Vec<TensorMeta>,
}

struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

pub struct VectorFont {
    tok_vocab: Vec<String>,
    letters: HashMap<String, u32>,
    arch: Arch,
    units: Units,
    t: HashMap<String, Tensor>,
    /// feats -> displacement from the previous cluster origin, in
    /// font units (for word-first contexts: the absolute origin).
    disp: HashMap<[u32; 5], (f64, f64)>,
    cache: std::cell::RefCell<HashMap<[u32; 5], Rc<BezPath>>>,
}

/// One laid-out cluster, origin in pixels (y-down, baseline y = 0).
#[derive(Clone)]
pub struct VCluster {
    pub letters: String,
    pub feats: [u32; 5],
    pub ox: f64,
    pub oy: f64,
}

pub fn is_vector_font(bytes: &[u8]) -> bool {
    if bytes.len() < 8 || &bytes[..4] != b"NTF0" {
        return false;
    }
    let hlen = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    serde_json::from_slice::<serde_json::Value>(&bytes[8..8 + hlen])
        .ok()
        .and_then(|v| v["format"].as_str().map(|s| s.starts_with("neuraltype-vector")))
        .unwrap_or(false)
}

impl VectorFont {
    pub fn load(bytes: &[u8]) -> Result<VectorFont, String> {
        if bytes.len() < 8 || &bytes[..4] != b"NTF0" {
            return Err("not a NeuralType file".into());
        }
        let hlen = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let header: Header =
            serde_json::from_slice(&bytes[8..8 + hlen]).map_err(|e| e.to_string())?;
        if header.format != "neuraltype-vector-v1" {
            return Err(format!("unsupported format {}", header.format));
        }
        let mut off = 8 + hlen;
        let mut t = HashMap::new();
        for tm in &header.tensors {
            let n: usize = tm.shape.iter().product();
            let end = off + n * 4;
            if end > bytes.len() {
                return Err("truncated vector font".into());
            }
            let data: Vec<f32> = bytes[off..end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            t.insert(tm.name.clone(), Tensor { shape: tm.shape.clone(), data });
            off = end;
        }
        let mut disp = HashMap::new();
        if let Some(dt) = t.get("disp.table") {
            for row in dt.data.chunks_exact(7) {
                let f = [
                    row[0] as u32,
                    row[1] as u32,
                    row[2] as u32,
                    row[3] as u32,
                    row[4] as u32,
                ];
                disp.insert(f, (row[5] as f64, row[6] as f64));
            }
        }
        let letters = header
            .letter_vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        Ok(VectorFont {
            tok_vocab: header.tok_vocab,
            letters,
            arch: header.arch,
            units: header.units,
            t,
            disp,
            cache: Default::default(),
        })
    }

    pub fn n_params(&self) -> usize {
        self.t.values().map(|t| t.data.len()).sum()
    }

    pub fn em_px(&self) -> f64 {
        self.units.em_px
    }

    pub fn alphabet(&self) -> String {
        let mut v: Vec<&String> = self.letters.keys().collect();
        v.sort();
        v.iter().filter(|s| s.chars().count() == 1).cloned().cloned().collect()
    }

    pub fn vocab_id(&self, s: &str) -> Option<u32> {
        self.letters.get(s).copied()
    }

    /// Outline for one letter-in-context: greedy decode, detokenize,
    /// cache. Font units, y-up, origin-relative.
    pub fn cluster_path(&self, feats: [u32; 5]) -> Rc<BezPath> {
        if let Some(p) = self.cache.borrow().get(&feats) {
            return p.clone();
        }
        let toks = self.decode(feats);
        let p = Rc::new(self.detokenize(&toks));
        self.cache.borrow_mut().insert(feats, p.clone());
        p
    }

    /// Segment a word into clusters (لا and word-الله fuses, matching
    /// the teacher) and chain origins from the displacement table.
    /// Origins are pixels, y-down, baseline y = 0.
    pub fn word_layout(&self, word: &str) -> Vec<VCluster> {
        let chars: Vec<char> = word.chars().collect();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        if chars == ['\u{627}', '\u{644}', '\u{644}', '\u{647}'] {
            ranges.push((0, 1));
            ranges.push((1, 4));
        } else {
            let mut i = 0;
            while i < chars.len() {
                if chars[i] == 'ل' && i + 1 < chars.len() && chars[i + 1] == 'ا' {
                    ranges.push((i, i + 2));
                    i += 2;
                } else {
                    ranges.push((i, i + 1));
                    i += 1;
                }
            }
        }
        let id_char = |c: Option<char>| -> u32 {
            c.and_then(|c| self.vocab_id(&c.to_string())).unwrap_or(0)
        };
        let scale = self.units.em_px / self.units.upm;
        let mut out = Vec::new();
        let mut ox_u = 0.0f64;
        let mut oy_u = 0.0f64;
        for (ci, &(a, b)) in ranges.iter().enumerate() {
            let letters: String = chars[a..b].iter().collect();
            let feats = [
                id_char(if a >= 2 { Some(chars[a - 2]) } else { None }),
                id_char(if a >= 1 { Some(chars[a - 1]) } else { None }),
                self.vocab_id(&letters).unwrap_or(0),
                id_char(chars.get(b).copied()),
                id_char(chars.get(b + 1).copied()),
            ];
            let (ddx, ddy) = self.disp.get(&feats).copied().unwrap_or((0.0, 0.0));
            if ci == 0 {
                ox_u = ddx;
                oy_u = ddy;
            } else {
                ox_u += ddx;
                oy_u += ddy;
            }
            out.push(VCluster {
                letters,
                feats,
                ox: ox_u * scale,
                oy: -oy_u * scale,
            });
        }
        out
    }

    /// One word as a single path: every cluster decoded, placed on
    /// the chain. Pixels, y-down, baseline y = 0.
    pub fn compose_word(&self, word: &str) -> (BezPath, Vec<VCluster>) {
        let clusters = self.word_layout(word);
        let scale = self.units.em_px / self.units.upm;
        let mut path = BezPath::new();
        for c in &clusters {
            let p = self.cluster_path(c.feats);
            let a = kurbo::Affine::new([scale, 0.0, 0.0, -scale, c.ox, c.oy]);
            for el in p.elements() {
                path.push(a * *el);
            }
        }
        (path, clusters)
    }

    /// Greedy decode with a per-layer KV cache. Returns token ids
    /// after BOS, up to and excluding EOS.
    fn decode(&self, feats: [u32; 5]) -> Vec<u32> {
        let a = &self.arch;
        let (d, hd) = (a.d, a.d / a.heads);
        let max_len = a.max_pos - a.prefix;
        let mut kcache: Vec<Vec<f32>> = vec![Vec::new(); a.layers];
        let mut vcache: Vec<Vec<f32>> = vec![Vec::new(); a.layers];
        let tok_emb = &self.t["tok_emb.weight"];
        let ctx_emb = &self.t["ctx_emb.weight"];
        let pos_emb = &self.t["pos_emb.weight"];
        let mut out = Vec::new();
        let bos = 1u32;
        let eos = 2u32;

        // one position through all layers; returns final hidden state
        let step = |x: &mut Vec<f32>,
                    kcache: &mut Vec<Vec<f32>>,
                    vcache: &mut Vec<Vec<f32>>| {
            for li in 0..self.arch.layers {
                let p = |n: &str| format!("b{li}.{n}");
                let h = ln(x, &self.t[&p("ln1.weight")], &self.t[&p("ln1.bias")]);
                let qkv = linear(&h, &self.t[&p("qkv.weight")], &self.t[&p("qkv.bias")]);
                kcache[li].extend_from_slice(&qkv[d..2 * d]);
                vcache[li].extend_from_slice(&qkv[2 * d..3 * d]);
                let t_len = kcache[li].len() / d;
                let mut att_out = vec![0.0f32; d];
                for hh in 0..self.arch.heads {
                    let qh = &qkv[hh * hd..(hh + 1) * hd];
                    let mut scores = Vec::with_capacity(t_len);
                    let mut smax = f32::NEG_INFINITY;
                    for ti in 0..t_len {
                        let kh = &kcache[li][ti * d + hh * hd..ti * d + (hh + 1) * hd];
                        let mut s = 0.0f32;
                        for (qi, ki) in qh.iter().zip(kh) {
                            s += qi * ki;
                        }
                        s /= (hd as f32).sqrt();
                        smax = smax.max(s);
                        scores.push(s);
                    }
                    let mut z = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - smax).exp();
                        z += *s;
                    }
                    for ti in 0..t_len {
                        let w = scores[ti] / z;
                        let vh = &vcache[li][ti * d + hh * hd..ti * d + (hh + 1) * hd];
                        for (o, vi) in att_out[hh * hd..(hh + 1) * hd].iter_mut().zip(vh) {
                            *o += w * vi;
                        }
                    }
                }
                let proj =
                    linear(&att_out, &self.t[&p("proj.weight")], &self.t[&p("proj.bias")]);
                for (xi, pi) in x.iter_mut().zip(&proj) {
                    *xi += pi;
                }
                let h = ln(x, &self.t[&p("ln2.weight")], &self.t[&p("ln2.bias")]);
                let mut f1 = linear(&h, &self.t[&p("fc1.weight")], &self.t[&p("fc1.bias")]);
                for v in f1.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
                let f2 = linear(&f1, &self.t[&p("fc2.weight")], &self.t[&p("fc2.bias")]);
                for (xi, fi) in x.iter_mut().zip(&f2) {
                    *xi += fi;
                }
            }
        };

        // prefix: the 5 context embeddings prime the cache
        for (p, &id) in feats.iter().enumerate() {
            let base = (id as usize).min(ctx_emb.shape[0] - 1) * d;
            let mut x: Vec<f32> = ctx_emb.data[base..base + d].to_vec();
            for (xi, pe) in x.iter_mut().zip(&pos_emb.data[p * d..(p + 1) * d]) {
                *xi += pe;
            }
            step(&mut x, &mut kcache, &mut vcache);
        }
        // decode from BOS
        let mut cur = bos;
        for i in 0..max_len {
            let base = (cur as usize).min(tok_emb.shape[0] - 1) * d;
            let mut x: Vec<f32> = tok_emb.data[base..base + d].to_vec();
            let p = a.prefix + i;
            for (xi, pe) in x.iter_mut().zip(&pos_emb.data[p * d..(p + 1) * d]) {
                *xi += pe;
            }
            step(&mut x, &mut kcache, &mut vcache);
            let h = ln(&x, &self.t["ln_f.weight"], &self.t["ln_f.bias"]);
            let logits = linear(&h, &self.t["head.weight"], &self.t["head.bias"]);
            let mut best = 0usize;
            for (j, v) in logits.iter().enumerate() {
                if *v > logits[best] {
                    best = j;
                }
            }
            let next = best as u32;
            if next == eos {
                break;
            }
            out.push(next);
            cur = next;
        }
        out
    }

    /// Tokens -> path in font units (y-up), origin-relative. Mirrors
    /// the trainer's detokenizer: chained deltas, |DMAX| continues.
    fn detokenize(&self, toks: &[u32]) -> BezPath {
        let q = self.units.q_units;
        let dmax = 63i64;
        let name = |t: u32| self.tok_vocab.get(t as usize).map(String::as_str).unwrap_or("");
        let mut i = 0usize;
        let (mut px, mut py) = (0.0f64, 0.0f64);
        let mut path = BezPath::new();
        let mut open = false;
        let read_delta = |i: &mut usize| -> f64 {
            let mut total = 0i64;
            while *i < toks.len() {
                let n = name(toks[*i]);
                if let Some(v) = n.strip_prefix('d').and_then(|s| s.parse::<i64>().ok()) {
                    *i += 1;
                    total += v;
                    if v.abs() < dmax {
                        break;
                    }
                } else {
                    break;
                }
            }
            total as f64 * q
        };
        while i < toks.len() {
            let n = name(toks[i]);
            i += 1;
            let count = match n {
                "M" => 1,
                "L" => 1,
                "Q" => 2,
                "C" => 3,
                "Z" => {
                    if open {
                        path.close_path();
                        open = false;
                    }
                    continue;
                }
                _ => continue,
            };
            let mut pts = Vec::with_capacity(count);
            for _ in 0..count {
                px += read_delta(&mut i);
                py += read_delta(&mut i);
                pts.push(kurbo::Point::new(px, py));
            }
            match n {
                "M" => {
                    if open {
                        path.close_path();
                    }
                    path.move_to(pts[0]);
                    open = true;
                }
                "L" if open => path.line_to(pts[0]),
                "Q" if open => path.quad_to(pts[0], pts[1]),
                "C" if open => path.curve_to(pts[0], pts[1], pts[2]),
                _ => {}
            }
        }
        if open {
            path.close_path();
        }
        path
    }
}

fn ln(x: &[f32], w: &Tensor, b: &Tensor) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    x.iter()
        .zip(&w.data)
        .zip(&b.data)
        .map(|((v, wi), bi)| (v - mean) * inv * wi + bi)
        .collect()
}

fn linear(x: &[f32], w: &Tensor, b: &Tensor) -> Vec<f32> {
    // candle Linear: weight (out, in), y = W x + b
    let out_dim = w.shape[0];
    let in_dim = w.shape[1];
    debug_assert_eq!(x.len(), in_dim);
    let mut y = b.data.clone();
    for o in 0..out_dim {
        let row = &w.data[o * in_dim..(o + 1) * in_dim];
        let mut acc = 0.0f32;
        for (wi, xi) in row.iter().zip(x) {
            acc += wi * xi;
        }
        y[o] += acc;
    }
    y
}
