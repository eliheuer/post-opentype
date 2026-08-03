//! ntf-train-vec: the vector-track prototype (docs/VECTOR.md).
//!
//! A small decoder-only transformer that emits outline tokens for a
//! letter-in-context cluster, trained on the teacher's own bezier
//! sequences (data/vec-gulzar, from scripts/vec_tokenizer.py). This
//! races the field model inside the same harness; nothing here
//! commits the project to the vector direction.
//!
//! Usage:
//!   ntf-train-vec <vec-dir> <out-dir> [epochs]
//!   ntf-train-vec sample <out-dir> <vec-dir> <word>
//!
//! Env: NTF_LR (3e-4), NTF_BS (64), NTF_MAX_ROWS (debug subset).
//!
//! The log format matches ntf-train so ntf-dash charts it: the
//! "val mse" slot carries validation cross-entropy and the "val IoU"
//! slot carries next-token accuracy.

use candle_core::{DType, Device, Tensor};
use candle_nn::ops;
use candle_nn::Optimizer;
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder, VarMap};
use std::collections::HashMap;

const D: usize = 192;
const LAYERS: usize = 4;
const HEADS: usize = 4;
const FFN: usize = 768;
const PREFIX: usize = 5;
const DMAX: i64 = 63;

#[derive(serde::Deserialize)]
struct Row {
    letters: String,
    prev2: Option<String>,
    prev: Option<String>,
    next: Option<String>,
    next2: Option<String>,
    tokens: Vec<u32>,
}

struct Dataset {
    letter_vocab: Vec<String>,
    tok_vocab: Vec<String>,
    ctx: Vec<[u32; 5]>,
    seqs: Vec<Vec<u32>>,
    max_len: usize,
}

fn load(vec_dir: &str) -> Dataset {
    let tok_vocab: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(format!("{vec_dir}/vocab.json")).unwrap())
            .unwrap();
    let mut letter_vocab: Vec<String> = vec!["<none>".into()];
    let mut lmap: HashMap<String, u32> = HashMap::new();
    lmap.insert("<none>".into(), 0);
    let mut id_of = |s: Option<&String>, v: &mut Vec<String>, m: &mut HashMap<String, u32>| -> u32 {
        match s {
            None => 0,
            Some(s) => {
                if let Some(&i) = m.get(s) {
                    i
                } else {
                    let i = v.len() as u32;
                    v.push(s.clone());
                    m.insert(s.clone(), i);
                    i
                }
            }
        }
    };
    let max_rows: usize = std::env::var("NTF_MAX_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let mut ctx = Vec::new();
    let mut seqs = Vec::new();
    let mut max_len = 0usize;
    let text = std::fs::read_to_string(format!("{vec_dir}/sequences.jsonl")).unwrap();
    for line in text.lines().take(max_rows) {
        let r: Row = serde_json::from_str(line).unwrap();
        let f = [
            id_of(r.prev2.as_ref(), &mut letter_vocab, &mut lmap),
            id_of(r.prev.as_ref(), &mut letter_vocab, &mut lmap),
            id_of(Some(&r.letters), &mut letter_vocab, &mut lmap),
            id_of(r.next.as_ref(), &mut letter_vocab, &mut lmap),
            id_of(r.next2.as_ref(), &mut letter_vocab, &mut lmap),
        ];
        max_len = max_len.max(r.tokens.len());
        ctx.push(f);
        seqs.push(r.tokens);
    }
    Dataset { letter_vocab, tok_vocab, ctx, seqs, max_len }
}

struct Block {
    ln1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl Block {
    fn new(vb: &VarBuilder) -> candle_core::Result<Self> {
        Ok(Block {
            ln1: layer_norm(D, 1e-5, vb.pp("ln1"))?,
            qkv: linear(D, 3 * D, vb.pp("qkv"))?,
            proj: linear(D, D, vb.pp("proj"))?,
            ln2: layer_norm(D, 1e-5, vb.pp("ln2"))?,
            fc1: linear(D, FFN, vb.pp("fc1"))?,
            fc2: linear(FFN, D, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let hd = D / HEADS;
        let h = self.ln1.forward(x)?;
        let qkv = self.qkv.forward(&h)?; // [b,t,3D]
        let q = qkv.narrow(2, 0, D)?;
        let k = qkv.narrow(2, D, D)?;
        let v = qkv.narrow(2, 2 * D, D)?;
        let shape = |z: &Tensor| -> candle_core::Result<Tensor> {
            z.reshape((b, t, HEADS, hd))?.transpose(1, 2)?.reshape((b * HEADS, t, hd))
        };
        let q = shape(&q)?;
        let k = shape(&k)?;
        let v = shape(&v)?;
        let att = (q.matmul(&k.transpose(1, 2)?)? / (hd as f64).sqrt())?;
        let att = att.broadcast_add(mask)?; // causal -inf above diagonal
        let att = ops::softmax(&att, 2)?;
        let out = att.matmul(&v)?; // [b*H, t, hd]
        let out = out
            .reshape((b, HEADS, t, hd))?
            .transpose(1, 2)?
            .reshape((b, t, D))?;
        let x = (x + self.proj.forward(&out)?)?;
        let h = self.ln2.forward(&x)?;
        let m = self.fc2.forward(&self.fc1.forward(&h)?.relu()?)?;
        x + m
    }
}

struct Model {
    tok_emb: Embedding,
    ctx_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    head: Linear,
}

impl Model {
    fn new(vb: &VarBuilder, tok_vocab: usize, letter_vocab: usize, max_pos: usize) -> candle_core::Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..LAYERS {
            blocks.push(Block::new(&vb.pp(format!("b{i}")))?);
        }
        Ok(Model {
            tok_emb: embedding(tok_vocab, D, vb.pp("tok_emb"))?,
            ctx_emb: embedding(letter_vocab, D, vb.pp("ctx_emb"))?,
            pos_emb: embedding(max_pos, D, vb.pp("pos_emb"))?,
            blocks,
            ln_f: layer_norm(D, 1e-5, vb.pp("ln_f"))?,
            head: linear(D, tok_vocab, vb.pp("head"))?,
        })
    }

    /// ctx [B,5] letter ids; toks [B,T] token ids. Returns logits [B, 5+T, V].
    fn forward(&self, ctx: &Tensor, toks: &Tensor, mask: &Tensor, device: &Device) -> candle_core::Result<Tensor> {
        let (b, t) = toks.dims2()?;
        let c = self.ctx_emb.forward(ctx)?; // [B,5,D]
        let e = self.tok_emb.forward(toks)?; // [B,T,D]
        let x = Tensor::cat(&[&c, &e], 1)?; // [B,5+T,D]
        let total = PREFIX + t;
        let pos_ids = Tensor::arange(0u32, total as u32, device)?
            .unsqueeze(0)?
            .broadcast_as((b, total))?;
        let x = (x + self.pos_emb.forward(&pos_ids)?)?;
        let mut x = x;
        for blk in &self.blocks {
            x = blk.forward(&x, mask)?;
        }
        self.head.forward(&self.ln_f.forward(&x)?)
    }
}

fn causal_mask(t: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mut m = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            m[i * t + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(m, (1, t, t), device)?.to_dtype(DType::F32)
}

/// Masked cross-entropy + accuracy over next-token targets.
/// Logits [B,5+T,V]; positions PREFIX-1 .. PREFIX+T-2 predict
/// toks[.., 1..T]; pads (id 0) are masked out.
fn loss_and_acc(
    logits: &Tensor,
    toks: &Tensor,
    device: &Device,
) -> candle_core::Result<(Tensor, f64, f64)> {
    let (b, t) = toks.dims2()?;
    let v = logits.dim(2)?;
    let pred = logits.narrow(1, PREFIX - 1, t)?; // predicts toks[0..t]
    // target toks[0..t] but position 0 target is BOS (constant) — keep it,
    // mask pads only
    let flat_logits = pred.reshape((b * t, v))?;
    let flat_targets = toks.reshape((b * t,))?;
    let logp = ops::log_softmax(&flat_logits, 1)?;
    let picked = logp.gather(&flat_targets.unsqueeze(1)?, 1)?.squeeze(1)?;
    let mask = flat_targets.ne(0u32)?.to_dtype(DType::F32)?;
    let n = mask.sum_all()?.to_scalar::<f32>()? as f64;
    let loss = (picked.neg()? * &mask)?.sum_all()? / n;
    let am = flat_logits.argmax(1)?.to_dtype(DType::U32)?;
    let correct = (am.eq(&flat_targets)?.to_dtype(DType::F32)? * &mask)?
        .sum_all()?
        .to_scalar::<f32>()? as f64;
    let _ = device;
    Ok((loss?, correct / n, n))
}

fn batches(order: &[usize], bs: usize) -> Vec<Vec<usize>> {
    order.chunks(bs).map(|c| c.to_vec()).collect()
}

fn pad_batch(ds: &Dataset, idx: &[usize], device: &Device) -> candle_core::Result<(Tensor, Tensor)> {
    let t = idx.iter().map(|&i| ds.seqs[i].len()).max().unwrap_or(1);
    let mut toks = vec![0u32; idx.len() * t];
    let mut ctx = vec![0u32; idx.len() * PREFIX];
    for (r, &i) in idx.iter().enumerate() {
        ctx[r * PREFIX..(r + 1) * PREFIX].copy_from_slice(&ds.ctx[i]);
        for (c, &tk) in ds.seqs[i].iter().enumerate() {
            toks[r * t + c] = tk;
        }
    }
    Ok((
        Tensor::from_vec(ctx, (idx.len(), PREFIX), device)?,
        Tensor::from_vec(toks, (idx.len(), t), device)?,
    ))
}

fn main() -> candle_core::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("sample") {
        return sample(
            args.get(1).expect("usage: ntf-train-vec sample <out-dir> <vec-dir> <word>"),
            args.get(2).expect("vec dir"),
            args.get(3).expect("word"),
        );
    }
    let vec_dir = args.first().expect("usage: ntf-train-vec <vec-dir> <out-dir> [epochs]");
    let out_dir = args.get(1).expect("out dir");
    let epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    std::fs::create_dir_all(out_dir).ok();

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;
    println!("device: {device:?}");

    let ds = load(vec_dir);
    let n = ds.seqs.len();
    println!(
        "rows: {} train, {} val",
        n - n / 20,
        n / 20
    );
    std::fs::write(
        format!("{out_dir}/letter-vocab.json"),
        serde_json::to_string(&ds.letter_vocab).unwrap(),
    )
    .unwrap();

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Model::new(&vb, ds.tok_vocab.len(), ds.letter_vocab.len(), PREFIX + ds.max_len)?;
    let ckpt = format!("{out_dir}/checkpoint.safetensors");
    if std::path::Path::new(&ckpt).exists() {
        varmap.load(&ckpt)?;
        println!("resumed from {ckpt}");
    }
    let nparams: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("model: {nparams} params ({:.1} MB f32)", nparams as f64 * 4.0 / 1e6);
    let lr: f64 = std::env::var("NTF_LR").ok().and_then(|v| v.parse().ok()).unwrap_or(3e-4);
    println!("lr: {lr}");
    let bs: usize = std::env::var("NTF_BS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    println!("batch size: {bs}");
    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), lr)?;

    // sort by length so batches pad minimally; shuffle batch order
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| ds.seqs[i].len());
    let val_idx: Vec<usize> = order.iter().copied().step_by(20).collect();
    let train_idx: Vec<usize> =
        order.iter().copied().enumerate().filter(|(k, _)| k % 20 != 0).map(|(_, i)| i).collect();
    let mut train_batches = batches(&train_idx, bs);
    let val_batches = batches(&val_idx, bs);

    let mut rng: u64 = 0x9e3779b97f4a7c15;
    for epoch in 1..=epochs {
        // xorshift shuffle of batch order
        for i in (1..train_batches.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            train_batches.swap(i, (rng as usize) % (i + 1));
        }
        let t0 = std::time::Instant::now();
        let mut loss_sum = 0.0f64;
        let mut nb = 0usize;
        for idx in &train_batches {
            let (ctx, toks) = pad_batch(&ds, idx, &device)?;
            let t = toks.dim(1)?;
            let mask = causal_mask(PREFIX + t, &device)?;
            let logits = model.forward(&ctx, &toks, &mask, &device)?;
            let (loss, _, _) = loss_and_acc(&logits, &toks, &device)?;
            opt.backward_step(&loss)?;
            loss_sum += loss.to_scalar::<f32>()? as f64;
            nb += 1;
        }
        let mut vl = 0.0f64;
        let mut vacc = 0.0f64;
        let mut vn = 0.0f64;
        for idx in &val_batches {
            let (ctx, toks) = pad_batch(&ds, idx, &device)?;
            let t = toks.dim(1)?;
            let mask = causal_mask(PREFIX + t, &device)?;
            let logits = model.forward(&ctx, &toks, &mask, &device)?;
            let (loss, acc, count) = loss_and_acc(&logits, &toks, &device)?;
            vl += loss.to_scalar::<f32>()? as f64 * count;
            vacc += acc * count;
            vn += count;
        }
        // log format matches ntf-train for ntf-dash: "val mse" = val
        // cross-entropy, "val IoU" = next-token accuracy
        println!(
            "epoch {epoch:3}  train loss {:.5}  val mse {:.5}  val IoU {:.4}  ({:.0}s)",
            loss_sum / nb.max(1) as f64,
            vl / vn.max(1.0),
            vacc / vn.max(1.0),
            t0.elapsed().as_secs_f64(),
        );
        varmap.save(&ckpt)?;
    }
    Ok(())
}

/// Greedy-decode the clusters of `word` and write SVGs to build/.
fn sample(out_dir: &str, vec_dir: &str, word: &str) -> candle_core::Result<()> {
    let device = Device::Cpu;
    let ds = load(vec_dir);
    let letter_vocab: Vec<String> = serde_json::from_str(
        &std::fs::read_to_string(format!("{out_dir}/letter-vocab.json")).unwrap(),
    )
    .unwrap();
    let lmap: HashMap<&str, u32> =
        letter_vocab.iter().enumerate().map(|(i, s)| (s.as_str(), i as u32)).collect();
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Model::new(&vb, ds.tok_vocab.len(), letter_vocab.len(), PREFIX + ds.max_len)?;
    varmap.load(format!("{out_dir}/checkpoint.safetensors"))?;

    // cluster the word (لا and word-الله fuses, matching the engine)
    let chars: Vec<char> = word.chars().collect();
    let mut cl: Vec<String> = Vec::new();
    if chars == ['\u{627}', '\u{644}', '\u{644}', '\u{647}'] {
        cl.push("\u{627}".into());
        cl.push("\u{644}\u{644}\u{647}".into());
    } else {
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == 'ل' && i + 1 < chars.len() && chars[i + 1] == 'ا' {
                cl.push("لا".into());
                i += 2;
            } else {
                cl.push(chars[i].to_string());
                i += 1;
            }
        }
    }
    let get = |k: i64| -> u32 {
        if k < 0 || k as usize >= cl.len() {
            0
        } else {
            *lmap.get(cl[k as usize].as_str()).unwrap_or(&0)
        }
    };
    let bos = 1u32;
    let eos = 2u32;
    for (k, letters) in cl.iter().enumerate() {
        let ki = k as i64;
        let feats = [get(ki - 2), get(ki - 1), get(ki), get(ki + 1), get(ki + 2)];
        let ctx = Tensor::from_vec(feats.to_vec(), (1, PREFIX), &device)?;
        let mut toks: Vec<u32> = vec![bos];
        for _ in 0..ds.max_len {
            let t = Tensor::from_vec(toks.clone(), (1, toks.len()), &device)?;
            let mask = causal_mask(PREFIX + toks.len(), &device)?;
            let logits = model.forward(&ctx, &t, &mask, &device)?;
            let last = logits.narrow(1, PREFIX + toks.len() - 1, 1)?.squeeze(1)?.squeeze(0)?;
            let next = last.argmax(0)?.to_scalar::<u32>()?;
            toks.push(next);
            if next == eos {
                break;
            }
        }
        let svg = detokenize(&toks, &ds.tok_vocab);
        let path = format!("build/vec-{letters}-{k}.svg");
        std::fs::create_dir_all("build").ok();
        std::fs::write(
            &path,
            format!(
                "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"-800 -1200 2000 2000\">\
                 <g transform=\"scale(1,-1)\"><path d=\"{svg}\" fill=\"#2aa35f\"/></g></svg>"
            ),
        )
        .unwrap();
        println!("{letters}: {} tokens -> {path}", toks.len());
    }
    Ok(())
}

/// Tokens -> absolute SVG path (font units, origin-relative).
fn detokenize(toks: &[u32], vocab: &[String]) -> String {
    let q = 2.0f64; // matches the tokenizer's q
    let mut i = 0usize;
    let mut px = 0.0f64;
    let mut py = 0.0f64;
    let mut out = String::new();
    let name = |t: u32| vocab.get(t as usize).map(String::as_str).unwrap_or("");
    let read_delta = |i: &mut usize| -> f64 {
        let mut total = 0i64;
        while *i < toks.len() {
            let n = name(toks[*i]);
            if let Some(v) = n.strip_prefix('d').and_then(|s| s.parse::<i64>().ok()) {
                *i += 1;
                total += v;
                if v.abs() < DMAX {
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
                out.push_str("Z ");
                continue;
            }
            _ => continue,
        };
        out.push_str(n);
        for _ in 0..count {
            px += read_delta(&mut i);
            py += read_delta(&mut i);
            out.push_str(&format!(" {px:.1} {py:.1}"));
        }
        out.push(' ');
    }
    out
}
