//! ntf-train: train the field model for a distilled font (candle).
//!
//! Step 3 of the TTF→NTF conversion (docs/DISTILL.md). Training uses
//! candle so it can run on CPU (default), Apple Accelerate
//! (--features accelerate), Metal (--features metal), or CUDA
//! (--features cuda). Only training uses a framework: inference stays
//! hand-rolled in neuraltype-core, and the exported .ntf carries raw
//! weights.
//!
//! Model: five context embeddings (prev2, prev, letter, next, next2)
//! → MLP latent → deconvolution decoder → SDF field on the shared
//! cluster canvas, plus a small head for the displacement to the next
//! cluster origin (the cascade).
//!
//! Usage: ntf-train <fields-dir> <out-dir> [epochs]

mod export;

use candle_core::{DType, Device, Tensor};
use candle_nn::{
    conv_transpose2d, embedding, linear, ConvTranspose2dConfig, Module, Optimizer, VarBuilder,
    VarMap,
};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Row {
    letters: String,
    prev: Option<char>,
    next: Option<char>,
    #[serde(default)]
    prev2: Option<char>,
    #[serde(default)]
    next2: Option<char>,
    shape: usize,
    ddx: Option<i32>,
    ddy: Option<i32>,
}

struct Dataset {
    /// Feature rows: [prev2, prev, letter, next, next2] vocab ids.
    feats: Vec<[u32; 5]>,
    shape_ids: Vec<usize>,
    /// Displacement targets in em units; NaN when absent.
    disp: Vec<[f32; 2]>,
    vocab: Vec<String>,
    fields: Vec<u8>,
    w: usize,
    h: usize,
    n_shapes: usize,
}

fn load(fields_dir: &str) -> Dataset {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{fields_dir}/fields-meta.json")).unwrap(),
    )
    .unwrap();
    let (w, h) = (
        meta["w"].as_u64().unwrap() as usize,
        meta["h"].as_u64().unwrap() as usize,
    );
    let upm = meta["upm"].as_f64().unwrap() as f32;
    let n_shapes = meta["shapes"].as_u64().unwrap() as usize;
    let fields = std::fs::read(format!("{fields_dir}/fields.bin")).unwrap();
    assert_eq!(fields.len(), n_shapes * w * h);

    // Vocabulary: id 0 = none/boundary; then every letters-string and char.
    let mut vocab_map: HashMap<String, u32> = HashMap::new();
    let mut vocab: Vec<String> = vec!["<none>".into()];
    vocab_map.insert("<none>".into(), 0);
    let mut id_of = |s: String, vocab: &mut Vec<String>, m: &mut HashMap<String, u32>| -> u32 {
        if let Some(&i) = m.get(&s) {
            return i;
        }
        let i = vocab.len() as u32;
        vocab.push(s.clone());
        m.insert(s, i);
        i
    };

    // Dedupe rows by feature tuple; keep the modal shape id and the
    // mean displacement (documented ambiguity ceiling).
    #[derive(Default)]
    struct Acc {
        shapes: HashMap<usize, usize>,
        dsum: [f64; 2],
        dn: usize,
    }
    let mut by_feat: HashMap<[u32; 5], Acc> = HashMap::new();
    let text = std::fs::read_to_string(format!("{fields_dir}/dataset.jsonl")).unwrap();
    for line in text.lines() {
        let r: Row = serde_json::from_str(line).unwrap();
        let f = [
            r.prev2.map_or(0, |c| id_of(c.to_string(), &mut vocab, &mut vocab_map)),
            r.prev.map_or(0, |c| id_of(c.to_string(), &mut vocab, &mut vocab_map)),
            id_of(r.letters.clone(), &mut vocab, &mut vocab_map),
            r.next.map_or(0, |c| id_of(c.to_string(), &mut vocab, &mut vocab_map)),
            r.next2.map_or(0, |c| id_of(c.to_string(), &mut vocab, &mut vocab_map)),
        ];
        let acc = by_feat.entry(f).or_default();
        *acc.shapes.entry(r.shape).or_default() += 1;
        if let (Some(dx), Some(dy)) = (r.ddx, r.ddy) {
            acc.dsum[0] += dx as f64 / upm as f64;
            acc.dsum[1] += dy as f64 / upm as f64;
            acc.dn += 1;
        }
    }
    let ambiguous = by_feat.values().filter(|a| a.shapes.len() > 1).count();
    println!(
        "{} unique context tuples ({} shape-ambiguous, {:.2}%), vocab {}",
        by_feat.len(),
        ambiguous,
        100.0 * ambiguous as f64 / by_feat.len() as f64,
        vocab.len()
    );

    let mut feats = Vec::new();
    let mut shape_ids = Vec::new();
    let mut disp = Vec::new();
    for (f, acc) in by_feat {
        let modal = acc.shapes.iter().max_by_key(|(_, n)| **n).unwrap().0;
        feats.push(f);
        shape_ids.push(*modal);
        if acc.dn > 0 {
            disp.push([
                (acc.dsum[0] / acc.dn as f64) as f32,
                (acc.dsum[1] / acc.dn as f64) as f32,
            ]);
        } else {
            disp.push([f32::NAN, f32::NAN]);
        }
    }
    Dataset { feats, shape_ids, disp, vocab, fields, w, h, n_shapes }
}

struct Model {
    emb: candle_nn::Embedding,
    l1: candle_nn::Linear,
    l2: candle_nn::Linear,
    disp: candle_nn::Linear,
    deconvs: Vec<candle_nn::ConvTranspose2d>,
    h: usize,
    w: usize,
}

const EMB: usize = 24;
const LATENT: usize = 256;
const C0: usize = 128;
const GRID0: (usize, usize) = (7, 5); // grows ×2 five times → (224, 160)

impl Model {
    fn new(vb: &VarBuilder, vocab: usize, h: usize, w: usize) -> candle_core::Result<Self> {
        let emb = embedding(vocab, EMB, vb.pp("emb"))?;
        let l1 = linear(5 * EMB, LATENT, vb.pp("l1"))?;
        let l2 = linear(LATENT, C0 * GRID0.0 * GRID0.1, vb.pp("l2"))?;
        let disp = linear(LATENT, 2, vb.pp("disp"))?;
        let chans = [C0, 64, 32, 16, 8, 1];
        let cfg = ConvTranspose2dConfig { padding: 1, output_padding: 0, stride: 2, dilation: 1 };
        let mut deconvs = Vec::new();
        for i in 0..5 {
            deconvs.push(conv_transpose2d(
                chans[i],
                chans[i + 1],
                4,
                cfg,
                vb.pp(format!("d{i}")),
            )?);
        }
        Ok(Model { emb, l1, l2, disp, deconvs, h, w })
    }

    /// Returns (field [B,1,h,w], displacement [B,2], latent).
    fn forward(&self, feats: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let b = feats.dim(0)?;
        let e = self.emb.forward(feats)?.reshape((b, 5 * EMB))?;
        let z = self.l1.forward(&e)?.relu()?;
        let disp = self.disp.forward(&z)?;
        let x = self.l2.forward(&z)?.relu()?;
        let mut x = x.reshape((b, C0, GRID0.0, GRID0.1))?;
        for (i, d) in self.deconvs.iter().enumerate() {
            x = d.forward(&x)?;
            if i + 1 < self.deconvs.len() {
                x = x.relu()?;
            }
        }
        // (B,1,224,160) → crop to (h,w)
        let x = x.narrow(2, 0, self.h)?.narrow(3, 0, self.w)?;
        Ok((x, disp))
    }
}

fn main() -> candle_core::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("export") {
        export::export(
            args.get(1).expect("usage: ntf-train export <train-dir> <fields-dir> <style> <out.ntf>"),
            args.get(2).expect("fields dir"),
            args.get(3).expect("style"),
            args.get(4).expect("out path"),
        );
        return Ok(());
    }
    let fields_dir = args.first().expect("usage: ntf-train <fields-dir> <out-dir> [epochs]");
    let out_dir = args.get(1).expect("usage: ntf-train <fields-dir> <out-dir> [epochs]");
    let epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    std::fs::create_dir_all(out_dir).unwrap();

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else if candle_core::utils::metal_is_available() {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };
    println!("device: {device:?}");

    let ds = load(fields_dir);
    let n = ds.feats.len();
    let (h, w) = (ds.h, ds.w);

    // All field targets on device once: [shapes, h*w] in [-1, 1].
    let fields_f32: Vec<f32> = ds.fields.iter().map(|&v| (v as f32 - 128.0) / 127.0).collect();
    let fields_t = Tensor::from_vec(fields_f32, (ds.n_shapes, h * w), &device)?;

    // Deterministic split: every 20th row is validation.
    let val_idx: Vec<usize> = (0..n).filter(|i| i % 20 == 0).collect();
    let train_idx: Vec<usize> = (0..n).filter(|i| i % 20 != 0).collect();
    println!("rows: {} train, {} val", train_idx.len(), val_idx.len());

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Model::new(&vb, ds.vocab.len(), h, w)?;
    let nparams: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("model: {nparams} params ({:.1} MB f32)", nparams as f64 * 4.0 / 1e6);

    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), 3e-4)?;

    let feats_of = |idx: &[usize]| -> candle_core::Result<Tensor> {
        let flat: Vec<u32> = idx.iter().flat_map(|&i| ds.feats[i]).collect();
        Tensor::from_vec(flat, (idx.len(), 5), &device)
    };
    let targets_of = |idx: &[usize]| -> candle_core::Result<Tensor> {
        let ids: Vec<u32> = idx.iter().map(|&i| ds.shape_ids[i] as u32).collect();
        let ids = Tensor::from_vec(ids, idx.len(), &device)?;
        fields_t.index_select(&ids, 0)?.reshape((idx.len(), 1, h, w))
    };
    let disp_of = |idx: &[usize]| -> (Tensor, Tensor) {
        let mut vals = Vec::with_capacity(idx.len() * 2);
        let mut mask = Vec::with_capacity(idx.len() * 2);
        for &i in idx {
            let d = ds.disp[i];
            let ok = !d[0].is_nan();
            vals.extend([if ok { d[0] } else { 0.0 }, if ok { d[1] } else { 0.0 }]);
            mask.extend([if ok { 1.0f32 } else { 0.0 }, if ok { 1.0 } else { 0.0 }]);
        }
        (
            Tensor::from_vec(vals, (idx.len(), 2), &device).unwrap(),
            Tensor::from_vec(mask, (idx.len(), 2), &device).unwrap(),
        )
    };

    const BS: usize = 128;
    let mut order: Vec<usize> = train_idx.clone();
    let mut rng_state = 0x9e3779b97f4a7c15u64;
    let mut shuffle = |v: &mut Vec<usize>| {
        for i in (1..v.len()).rev() {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            v.swap(i, (rng_state as usize) % (i + 1));
        }
    };

    for epoch in 1..=epochs {
        shuffle(&mut order);
        let mut loss_sum = 0.0f64;
        let mut nb = 0usize;
        let t0 = std::time::Instant::now();
        for chunk in order.chunks(BS) {
            let feats = feats_of(chunk)?;
            let target = targets_of(chunk)?;
            let (dtgt, dmask) = disp_of(chunk);
            let (pred, dpred) = model.forward(&feats)?;
            let field_loss = (pred.sub(&target))?.sqr()?.mean_all()?;
            let disp_loss = ((dpred.sub(&dtgt))?.sqr()? * &dmask)?.mean_all()?;
            let loss = (field_loss + (disp_loss * 0.1)?)?;
            opt.backward_step(&loss)?;
            loss_sum += loss.to_scalar::<f32>()? as f64;
            nb += 1;
        }
        // Validation: field MSE and IoU at the contour.
        let mut vmse = 0.0f64;
        let mut inter = 0.0f64;
        let mut union = 0.0f64;
        for chunk in val_idx.chunks(BS) {
            let feats = feats_of(chunk)?;
            let target = targets_of(chunk)?;
            let (pred, _) = model.forward(&feats)?;
            vmse += (pred.sub(&target))?.sqr()?.mean_all()?.to_scalar::<f32>()? as f64
                * chunk.len() as f64;
            let pi = pred.ge(0.0)?.to_dtype(DType::F32)?;
            let ti = target.ge(0.0)?.to_dtype(DType::F32)?;
            inter += (&pi * &ti)?.sum_all()?.to_scalar::<f32>()? as f64;
            union += (((&pi + &ti)? - (&pi * &ti)?)?).sum_all()?.to_scalar::<f32>()? as f64;
        }
        println!(
            "epoch {epoch:3}  train loss {:.5}  val mse {:.5}  val IoU {:.4}  ({:.0}s)",
            loss_sum / nb as f64,
            vmse / val_idx.len() as f64,
            inter / union,
            t0.elapsed().as_secs_f64()
        );
        varmap.save(format!("{out_dir}/checkpoint.safetensors"))?;
    }

    // Persist the vocabulary next to the checkpoint for export.
    std::fs::write(
        format!("{out_dir}/vocab.json"),
        serde_json::to_string(&ds.vocab).unwrap(),
    )
    .unwrap();
    println!("saved {out_dir}/checkpoint.safetensors");
    Ok(())
}
