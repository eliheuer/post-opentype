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
    /// Priority. Rows from a designer's labeled phrases carry 1 and
    /// replace teacher rows (0) that share their context.
    #[serde(default)]
    pri: u8,
    /// Cluster index in its word; 0 starts a word.
    #[serde(default)]
    index: usize,
}

/// A neighboring cluster in the word a context was first seen in: its
/// shape, and its origin relative to this one (whole pixels, y down).
#[derive(Clone, Copy)]
struct Nb {
    shape: usize,
    dx: i64,
    dy: i64,
}

struct Dataset {
    /// Feature rows: [prev2, prev, letter, next, next2] vocab ids.
    feats: Vec<[u32; 5]>,
    shape_ids: Vec<usize>,
    /// Displacement targets in em units; NaN when absent.
    disp: Vec<[f32; 2]>,
    /// True for rows that came from labeled phrases (pri > 0).
    hand: Vec<bool>,
    /// The clusters before and after, for stretch targets.
    nb: Vec<[Option<Nb>; 2]>,
    em_px: f32,
    spread_px: f32,
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
    let em_px = meta["em_px"].as_f64().unwrap() as f32;
    let spread_px = meta["spread_px"].as_f64().unwrap() as f32;
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
        pri: u8,
        /// Row order of the first occurrence, so the tuple list is
        /// deterministic.
        first: usize,
        /// Neighbors (previous, next) of the row that set the shape.
        nb: [Option<Nb>; 2],
        /// The shape that row had.
        nb_shape: Option<usize>,
    }
    let mut by_feat: HashMap<[u32; 5], Acc> = HashMap::new();
    let text = std::fs::read_to_string(format!("{fields_dir}/dataset.jsonl")).unwrap();
    // the previous line's cluster, when it is in the same word
    let mut last: Option<([u32; 5], usize)> = None;
    let px = |units: i32| (units as f32 / upm * em_px).round() as i64;
    for line in text.lines() {
        let r: Row = serde_json::from_str(line).unwrap();
        let f = [
            r.prev2.map_or(0, |c| id_of(c.to_string(), &mut vocab, &mut vocab_map)),
            r.prev.map_or(0, |c| id_of(c.to_string(), &mut vocab, &mut vocab_map)),
            id_of(r.letters.clone(), &mut vocab, &mut vocab_map),
            r.next.map_or(0, |c| id_of(c.to_string(), &mut vocab, &mut vocab_map)),
            r.next2.map_or(0, |c| id_of(c.to_string(), &mut vocab, &mut vocab_map)),
        ];
        // Link this cluster to the one before it in the word. Spaces
        // and word starts break the chain.
        let is_space = r.letters == " ";
        let link = if r.index > 0 && !is_space { last } else { None };
        last = if is_space { None } else { Some((f, r.shape)) };
        let n_seen = by_feat.len();
        let acc = by_feat.entry(f).or_insert_with(|| Acc { first: n_seen, ..Default::default() });
        if r.pri < acc.pri {
            continue;
        }
        if r.pri > acc.pri {
            // a labeled row replaces every teacher row for its context
            *acc = Acc { pri: r.pri, first: acc.first, ..Default::default() };
        } else if r.pri > 0 {
            // a later labeled instance of the same context: the first
            // one stands, no modal vote and no averaged displacement
            continue;
        }
        if acc.nb_shape.is_none() {
            acc.nb_shape = Some(r.shape);
        }
        if let (Some((pf, pshape)), Some(dx), Some(dy)) = (link, r.ddx, r.ddy) {
            // this cluster sits at (dx, -dy) px from the previous one
            let (dx, dy) = (px(dx), -px(dy));
            if acc.nb_shape == Some(r.shape) && acc.nb[0].is_none() {
                acc.nb[0] = Some(Nb { shape: pshape, dx: -dx, dy: -dy });
            }
            let shape = r.shape;
            if let Some(pacc) = by_feat.get_mut(&pf) {
                if pacc.nb_shape == Some(pshape) && pacc.nb[1].is_none() {
                    pacc.nb[1] = Some(Nb { shape, dx, dy });
                }
            }
        }
        let acc = by_feat.get_mut(&f).unwrap();
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
    let mut hand = Vec::new();
    let mut nb = Vec::new();
    let mut tuples: Vec<([u32; 5], Acc)> = by_feat.into_iter().collect();
    tuples.sort_by_key(|(_, a)| a.first);
    for (f, acc) in tuples {
        hand.push(acc.pri > 0);
        nb.push(acc.nb);
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
    Dataset { feats, shape_ids, disp, hand, nb, em_px, spread_px, vocab, fields, w, h, n_shapes }
}

struct Model {
    emb: candle_nn::Embedding,
    l1: candle_nn::Linear,
    /// Optional second hidden layer (NTF_DEEP=1).
    l1b: Option<candle_nn::Linear>,
    l2: candle_nn::Linear,
    disp: candle_nn::Linear,
    deconvs: Vec<candle_nn::ConvTranspose2d>,
    h: usize,
    w: usize,
}

/// Model dimensions, overridable per run. Raising the canvas
/// resolution alone does not buy fidelity: it grows l2 and the seed
/// grid while the shape code (LATENT) and the decoder's channel
/// widths stay put, so the same amount of learned shape is spread
/// over more pixels. Widen these instead.
fn envd(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
static EMB: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| envd("NTF_EMB", 24));
static LATENT: std::sync::LazyLock<usize> =
    std::sync::LazyLock::new(|| envd("NTF_LATENT", 256));
/// Deconvolution channel ladder, seed first and output last. Its
/// length sets the number of stride-2 stages, and so the seed grid.
static CHANS: std::sync::LazyLock<Vec<usize>> = std::sync::LazyLock::new(|| {
    std::env::var("NTF_CHANS")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect::<Vec<usize>>())
        .filter(|v| v.len() >= 2)
        .unwrap_or_else(|| vec![128, 64, 32, 16, 8, 1])
});
static C0: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| CHANS[0]);
/// Conditioning inputs after the embeddings: 4 (the pulls on the two
/// joins, see neuraltype_core::stretch) when NTF_STRETCH is set.
static COND: std::sync::LazyLock<usize> =
    std::sync::LazyLock::new(|| if envf("NTF_STRETCH", 0.0) > 0.0 { 4 } else { envd("NTF_COND", 0) });
static DEEP: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| envd("NTF_DEEP", 0) > 0);
fn envf(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Seed grid for the deconvolution stack: each stride-2 layer doubles
/// it, so the seed is the canvas dims divided by 2^stages, rounded
/// up. (7, 5) for the 64 px/em canvas with the default five stages.
fn grid0_for(h: usize, w: usize) -> (usize, usize) {
    let f = 1usize << (CHANS.len() - 1);
    ((h + f - 1) / f, (w + f - 1) / f)
}

impl Model {
    fn new(vb: &VarBuilder, vocab: usize, h: usize, w: usize) -> candle_core::Result<Self> {
        let g0 = grid0_for(h, w);
        let emb = embedding(vocab, *EMB, vb.pp("emb"))?;
        let l1 = linear(5 * *EMB + *COND, *LATENT, vb.pp("l1"))?;
        let l1b = if *DEEP { Some(linear(*LATENT, *LATENT, vb.pp("l1b"))?) } else { None };
        let l2 = linear(*LATENT, *C0 * g0.0 * g0.1, vb.pp("l2"))?;
        let disp = linear(*LATENT, 2, vb.pp("disp"))?;
        let chans = &*CHANS;
        let cfg = ConvTranspose2dConfig { padding: 1, output_padding: 0, stride: 2, dilation: 1 };
        let mut deconvs = Vec::new();
        for i in 0..chans.len() - 1 {
            deconvs.push(conv_transpose2d(
                chans[i],
                chans[i + 1],
                4,
                cfg,
                vb.pp(format!("d{i}")),
            )?);
        }
        Ok(Model { emb, l1, l1b, l2, disp, deconvs, h, w })
    }

    /// Returns (field [B,1,h,w], displacement [B,2], latent).
    fn forward(&self, feats: &Tensor, cond: Option<&Tensor>) -> candle_core::Result<(Tensor, Tensor)> {
        let b = feats.dim(0)?;
        let mut e = self.emb.forward(feats)?.reshape((b, 5 * *EMB))?;
        if *COND > 0 {
            let c = match cond {
                Some(c) => c.clone(),
                None => Tensor::zeros((b, *COND), DType::F32, feats.device())?,
            };
            e = Tensor::cat(&[&e, &c], 1)?;
        }
        let mut z = self.l1.forward(&e)?.relu()?;
        if let Some(l) = &self.l1b {
            z = l.forward(&z)?.relu()?;
        }
        let disp = self.disp.forward(&z)?;
        let x = self.l2.forward(&z)?.relu()?;
        let g0 = grid0_for(self.h, self.w);
        let mut x = x.reshape((b, *C0, g0.0, g0.1))?;
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

/// If the dataset vocabulary grew since the checkpoint was written
/// (corpus extension), expand emb.weight in place: keep the old rows,
/// add small random rows for the new tokens. Ids stay stable because
/// new corpus words append after the old ones.
fn expand_checkpoint_vocab(ckpt: &str, vocab_len: usize) -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let mut t = candle_core::safetensors::load(ckpt, &dev)?;
    let emb = t.get("emb.weight").expect("emb.weight in checkpoint").clone();
    let (rows, cols) = emb.dims2()?;
    if rows == vocab_len {
        return Ok(());
    }
    assert!(rows < vocab_len, "checkpoint vocab larger than dataset vocab");
    let extra = Tensor::randn(0f32, 0.02f32, (vocab_len - rows, cols), &dev)?;
    let expanded = Tensor::cat(&[&emb, &extra], 0)?;
    println!("expanded emb.weight: {rows} -> {vocab_len} rows");
    t.insert("emb.weight".to_string(), expanded);
    candle_core::safetensors::save(&t, ckpt)?;
    Ok(())
}

/// Bring a checkpoint up to this run's architecture without changing
/// what it computes: new conditioning inputs get zero weights, and a
/// new second hidden layer starts as the identity (its inputs are
/// already non-negative, so the ReLU after it changes nothing).
fn expand_checkpoint_arch(ckpt: &str) -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let mut t = candle_core::safetensors::load(ckpt, &dev)?;
    let mut changed = false;
    let l1 = t.get("l1.weight").expect("l1.weight in checkpoint").clone();
    let (rows, cols) = l1.dims2()?;
    let want = 5 * *EMB + *COND;
    if cols < want {
        let extra = Tensor::zeros((rows, want - cols), DType::F32, &dev)?;
        t.insert("l1.weight".to_string(), Tensor::cat(&[&l1, &extra], 1)?);
        println!("expanded l1.weight: {cols} -> {want} inputs");
        changed = true;
    }
    if *DEEP && !t.contains_key("l1b.weight") {
        t.insert("l1b.weight".to_string(), Tensor::eye(rows, DType::F32, &dev)?);
        t.insert("l1b.bias".to_string(), Tensor::zeros(rows, DType::F32, &dev)?);
        println!("added l1b as identity");
        changed = true;
    }
    if changed {
        candle_core::safetensors::save(&t, ckpt)?;
    }
    Ok(())
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
    let fields_t = Tensor::from_vec(fields_f32.clone(), (ds.n_shapes, h * w), &device)?;
    let field_of = |shape: usize| &fields_f32[shape * h * w..(shape + 1) * h * w];

    // Deterministic split: every 20th teacher row is validation.
    // Labeled rows always train: there are too few to hold any out.
    let val_idx: Vec<usize> = (0..n).filter(|&i| i % 20 == 0 && !ds.hand[i]).collect();
    let mut train_idx: Vec<usize> = (0..n).filter(|&i| i % 20 != 0 && !ds.hand[i]).collect();
    let hand_idx: Vec<usize> = (0..n).filter(|&i| ds.hand[i]).collect();
    println!("rows: {} train, {} val, {} labeled", train_idx.len(), val_idx.len(), hand_idx.len());

    // Oversample the long-word rows. A context window that spans
    // four or more letters cannot come from the combinatorial corpus
    // (singles, pairs, triples), so these rows belong to the
    // real-word extension (basmala, surahs), and they are a sliver
    // of the gradient. NTF_OVERSAMPLE=K repeats them K times per
    // epoch; the validation split is untouched.
    let os: usize =
        std::env::var("NTF_OVERSAMPLE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let lig = ds.vocab.iter().position(|s| s == "\u{644}\u{644}\u{647}").map(|i| i as u32);
    let long: Vec<usize> = train_idx
        .iter()
        .copied()
        .filter(|&i| {
            let f = ds.feats[i];
            (f[0] != 0 && f[3] != 0) || (f[1] != 0 && f[4] != 0) || Some(f[2]) == lig
        })
        .collect();
    if os > 1 {
        println!("oversampling {} long-word rows x{os}", long.len());
        for _ in 1..os {
            train_idx.extend_from_slice(&long);
        }
    }

    // A resumed run must keep the token ids its checkpoint was
    // trained with: the old vocabulary has to be a prefix of the new.
    if let Ok(old) = std::fs::read_to_string(format!("{out_dir}/vocab.json")) {
        let old: Vec<String> = serde_json::from_str(&old).unwrap();
        assert!(
            ds.vocab.len() >= old.len() && ds.vocab[..old.len()] == old[..],
            "dataset vocabulary does not extend the checkpoint's: {:?} vs {:?}",
            ds.vocab,
            old
        );
    }
    // Persist the vocabulary up front so mid-training checkpoints can
    // be exported.
    std::fs::write(
        format!("{out_dir}/vocab.json"),
        serde_json::to_string(&ds.vocab).unwrap(),
    )
    .unwrap();

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Model::new(&vb, ds.vocab.len(), h, w)?;
    // Resume: if a checkpoint already exists in out-dir, load it and
    // continue training from there (optimizer state starts fresh).
    let ckpt = format!("{out_dir}/checkpoint.safetensors");
    if std::path::Path::new(&ckpt).exists() {
        expand_checkpoint_vocab(&ckpt, ds.vocab.len())?;
        expand_checkpoint_arch(&ckpt)?;
        varmap.load(&ckpt)?;
        println!("resumed from {ckpt}");
    }
    let nparams: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("model: {nparams} params ({:.1} MB f32)", nparams as f64 * 4.0 / 1e6);

    let lr: f64 = std::env::var("NTF_LR").ok().and_then(|v| v.parse().ok()).unwrap_or(3e-4);
    println!("lr: {lr}");
    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), lr)?;

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

    // NTF_BS is the documented name; NTF_bs is what older runs set.
    let bs: usize = std::env::var("NTF_BS")
        .or_else(|_| std::env::var("NTF_bs"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    println!("batch size: {bs}");
    // Fine-tuning on labeled phrases: each epoch sees every labeled
    // row NTF_HAND_OS times plus NTF_REPLAY teacher rows drawn fresh,
    // so the model keeps the teacher's behavior for all other text.
    // NTF_REPLAY=0 (default) uses every teacher row each epoch.
    let replay = envd("NTF_REPLAY", 0);
    let hand_os = envd("NTF_HAND_OS", 1);
    if !hand_idx.is_empty() {
        println!(
            "labeled rows x{hand_os} per epoch, teacher replay {replay} + {} long-word rows",
            long.len()
        );
    }
    let mut order: Vec<usize> = Vec::new();
    let mut rng_state = 0x9e3779b97f4a7c15u64;
    let mut shuffle = |v: &mut Vec<usize>| {
        for i in (1..v.len()).rev() {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            v.swap(i, (rng_state as usize) % (i + 1));
        }
    };

    // Stretch training (NTF_STRETCH = share of focus rows pulled per
    // epoch). The focus rows are the labeled rows, the long-word
    // rows, and the clusters of the words in NTF_STRETCH_WORDS.
    let stretch = envf("NTF_STRETCH", 0.0);
    let stretch_max = envf("NTF_STRETCH_MAX", 0.8) * ds.em_px;
    let geom = neuraltype_core::stretch::Geometry { w, h, em_px: ds.em_px, spread_px: ds.spread_px };
    let mut focus = vec![false; n];
    for &i in hand_idx.iter().chain(&long) {
        focus[i] = true;
    }
    let tuple_of: HashMap<[u32; 5], usize> =
        ds.feats.iter().enumerate().map(|(i, f)| (*f, i)).collect();
    if let Ok(words) = std::env::var("NTF_STRETCH_WORDS") {
        let id = |s: &str| ds.vocab.iter().position(|v| v == s).map_or(0, |i| i as u32);
        for word in words.split(',').filter(|w| !w.is_empty()) {
            let chars: Vec<char> = word.chars().collect();
            for (a, b) in neuraltype_core::field_text::cluster_ranges(&chars) {
                let ch = |i: Option<usize>| i.and_then(|i| chars.get(i)).map_or(0, |c| id(&c.to_string()));
                let f = [
                    ch(a.checked_sub(2)),
                    ch(a.checked_sub(1)),
                    id(&chars[a..b].iter().collect::<String>()),
                    ch(Some(b)),
                    ch(Some(b + 1)),
                ];
                match tuple_of.get(&f) {
                    Some(&i) => focus[i] = true,
                    None => println!("stretch word {word}: a cluster is not in the dataset"),
                }
            }
        }
    }
    let focus_idx: Vec<usize> = (0..n).filter(|&i| focus[i] && i % 20 != 0 || focus[i] && ds.hand[i]).collect();
    let focus_os = envd("NTF_FOCUS_OS", 1);
    if stretch > 0.0 {
        println!(
            "stretch: {} focus rows x{focus_os}, share {stretch}, up to {:.0} px",
            focus_idx.len(),
            stretch_max
        );
    }
    let mut rng2 = 0x2545f4914f6cdd1du64;
    let mut rand01 = move || -> f32 {
        rng2 ^= rng2 << 13;
        rng2 ^= rng2 >> 7;
        rng2 ^= rng2 << 17;
        (rng2 >> 40) as f32 / (1u64 << 24) as f32
    };
    // A pulled target for row i: random pulls on the joins it has.
    let pulled_row = |i: usize, rand01: &mut dyn FnMut() -> f32| {
        use neuraltype_core::stretch::{pulled, Neighbor};
        let mut pull = |rand01: &mut dyn FnMut() -> f32| -> (f32, f32) {
            if rand01() < 0.35 {
                return (0.0, 0.0);
            }
            let dx = -(rand01() * stretch_max).round();
            let dy = if rand01() < 0.3 { ((rand01() - 0.5) * 0.24 * ds.em_px).round() } else { 0.0 };
            (dx, dy)
        };
        let nbs: Vec<Option<Neighbor>> = ds.nb[i]
            .iter()
            .map(|nb| nb.map(|nb| Neighbor { field: field_of(nb.shape), dx: nb.dx, dy: nb.dy }))
            .collect();
        let prev = nbs[0].as_ref().map(|nb| (nb, pull(rand01)));
        let next = nbs[1].as_ref().map(|nb| (nb, pull(rand01)));
        pulled(field_of(ds.shape_ids[i]), &geom, prev, next)
    };

    for epoch in 1..=epochs {
        shuffle(&mut train_idx);
        order.clear();
        let take = if replay == 0 { train_idx.len() } else { replay.min(train_idx.len()) };
        order.extend_from_slice(&train_idx[..take]);
        if replay != 0 {
            // A replay sample would almost never draw the long-word
            // rows, and they are the first thing the model forgets.
            order.extend_from_slice(&long);
        }
        for _ in 0..hand_os {
            order.extend_from_slice(&hand_idx);
        }
        if stretch > 0.0 {
            for _ in 0..focus_os {
                order.extend_from_slice(&focus_idx);
            }
        }
        shuffle(&mut order);
        let mut loss_sum = 0.0f64;
        let mut nb = 0usize;
        let t0 = std::time::Instant::now();
        for chunk in order.chunks(bs) {
            let feats = feats_of(chunk)?;
            let target = targets_of(chunk)?;
            let (dtgt, dmask) = disp_of(chunk);
            // Stretch: some rows train on a pulled version of their
            // field, with the pull as the model's extra input.
            let mut cond_t = None;
            let mut target = target;
            if stretch > 0.0 {
                let mut cond = vec![0.0f32; chunk.len() * 4];
                let mut flat: Option<Vec<f32>> = None;
                for (bi, &i) in chunk.iter().enumerate() {
                    if !focus[i] || rand01() >= stretch {
                        continue;
                    }
                    let (field, done) = pulled_row(i, &mut rand01);
                    cond[bi * 4..bi * 4 + 4].copy_from_slice(&done.cond(ds.em_px));
                    let flat = flat.get_or_insert_with(|| {
                        chunk.iter().flat_map(|&j| field_of(ds.shape_ids[j]).to_vec()).collect()
                    });
                    flat[bi * h * w..(bi + 1) * h * w].copy_from_slice(&field);
                }
                if let Some(flat) = flat {
                    target = Tensor::from_vec(flat, (chunk.len(), 1, h, w), &device)?;
                }
                cond_t = Some(Tensor::from_vec(cond, (chunk.len(), 4), &device)?);
            }
            let (pred, dpred) = model.forward(&feats, cond_t.as_ref())?;
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
        for chunk in val_idx.chunks(bs) {
            let feats = feats_of(chunk)?;
            let target = targets_of(chunk)?;
            let (pred, _) = model.forward(&feats, None)?;
            vmse += (pred.sub(&target))?.sqr()?.mean_all()?.to_scalar::<f32>()? as f64
                * chunk.len() as f64;
            let pi = pred.ge(0.0)?.to_dtype(DType::F32)?;
            let ti = target.ge(0.0)?.to_dtype(DType::F32)?;
            inter += (&pi * &ti)?.sum_all()?.to_scalar::<f32>()? as f64;
            union += (((&pi + &ti)? - (&pi * &ti)?)?).sum_all()?.to_scalar::<f32>()? as f64;
        }
        // Labeled rows: field IoU and displacement error (font units).
        let mut hand_note = String::new();
        if !hand_idx.is_empty() {
            let (mut hi, mut hu, mut de, mut dn) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for chunk in hand_idx.chunks(bs) {
                let feats = feats_of(chunk)?;
                let target = targets_of(chunk)?;
                let (dtgt, dmask) = disp_of(chunk);
                let (pred, dpred) = model.forward(&feats, None)?;
                let pi = pred.ge(0.0)?.to_dtype(DType::F32)?;
                let ti = target.ge(0.0)?.to_dtype(DType::F32)?;
                hi += (&pi * &ti)?.sum_all()?.to_scalar::<f32>()? as f64;
                hu += (((&pi + &ti)? - (&pi * &ti)?)?).sum_all()?.to_scalar::<f32>()? as f64;
                de += ((dpred.sub(&dtgt))?.abs()? * &dmask)?.sum_all()?.to_scalar::<f32>()? as f64;
                dn += dmask.sum_all()?.to_scalar::<f32>()? as f64;
            }
            hand_note = format!(
                "  labeled IoU {:.4}  disp err {:.1}u",
                hi / hu.max(1.0),
                1000.0 * de / dn.max(1.0)
            );
        }
        // Stretch: IoU against the geometric target, at a fixed pull
        // of half the maximum on the next join.
        if stretch > 0.0 && !focus_idx.is_empty() {
            use neuraltype_core::stretch::{pulled, Neighbor};
            let (mut si, mut su) = (0.0f64, 0.0f64);
            let rows: Vec<usize> =
                focus_idx.iter().copied().filter(|&i| ds.nb[i][1].is_some()).take(96).collect();
            for chunk in rows.chunks(bs) {
                let mut flat = Vec::with_capacity(chunk.len() * h * w);
                let mut cond = Vec::with_capacity(chunk.len() * 4);
                for &i in chunk {
                    let nb = ds.nb[i][1].unwrap();
                    let nb = Neighbor { field: field_of(nb.shape), dx: nb.dx, dy: nb.dy };
                    let d = (-(stretch_max * 0.5).round(), 0.0);
                    let (f, done) = pulled(field_of(ds.shape_ids[i]), &geom, None, Some((&nb, d)));
                    flat.extend(f);
                    cond.extend(done.cond(ds.em_px));
                }
                let target = Tensor::from_vec(flat, (chunk.len(), 1, h, w), &device)?;
                let cond = Tensor::from_vec(cond, (chunk.len(), 4), &device)?;
                let (pred, _) = model.forward(&feats_of(chunk)?, Some(&cond))?;
                let pi = pred.ge(0.0)?.to_dtype(DType::F32)?;
                let ti = target.ge(0.0)?.to_dtype(DType::F32)?;
                si += (&pi * &ti)?.sum_all()?.to_scalar::<f32>()? as f64;
                su += (((&pi + &ti)? - (&pi * &ti)?)?).sum_all()?.to_scalar::<f32>()? as f64;
            }
            hand_note += &format!("  stretch IoU {:.4}", si / su.max(1.0));
        }
        println!(
            "epoch {epoch:3}  train loss {:.5}  val mse {:.5}  val IoU {:.4}{hand_note}  ({:.0}s)",
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
