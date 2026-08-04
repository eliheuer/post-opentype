//! Export a trained field-model checkpoint to a .ntf file.
//!
//! Format "neuraltype-field-v1": the same container as v0 (magic,
//! u32 LE header length, JSON header padded to 4-byte alignment, raw
//! f32 LE weights) with a header that fully describes the
//! architecture and canvas, so the engine's hand-rolled inference can
//! run it without a framework. Weight order is fixed and listed in
//! the header's `tensors` field.

use std::io::Write as _;

pub fn export(train_dir: &str, fields_dir: &str, style: &str, out_path: &str) {
    let vocab: Vec<String> = serde_json::from_str(
        &std::fs::read_to_string(format!("{train_dir}/vocab.json")).unwrap(),
    )
    .unwrap();
    let fmeta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{fields_dir}/fields-meta.json")).unwrap(),
    )
    .unwrap();
    let bytes = std::fs::read(format!("{train_dir}/checkpoint.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();

    // Architecture is read back from the checkpoint's own tensor
    // shapes, so a run trained with different NTF_LATENT/NTF_CHANS
    // exports correctly without passing the env through again.
    let n_deconv = (0..).take_while(|i| st.tensor(&format!("d{i}.weight")).is_ok()).count();
    let emb_dim = st.tensor("emb.weight").unwrap().shape()[1];
    let latent = st.tensor("l1.weight").unwrap().shape()[0];
    let mut chans: Vec<usize> = vec![st.tensor("d0.weight").unwrap().shape()[0]];
    for i in 0..n_deconv {
        chans.push(st.tensor(&format!("d{i}.weight")).unwrap().shape()[1]);
    }
    let c0 = chans[0];
    let seed_div = 1usize << n_deconv;

    // Fixed serialization order; names match the trainer's VarBuilder.
    let mut order: Vec<String> = [
        "emb.weight",
        "l1.weight",
        "l1.bias",
        "l2.weight",
        "l2.bias",
        "disp.weight",
        "disp.bias",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for i in 0..n_deconv {
        order.push(format!("d{i}.weight"));
        order.push(format!("d{i}.bias"));
    }

    let mut tensors_meta = Vec::new();
    let mut blob: Vec<u8> = Vec::new();
    for name in &order {
        let t = st
            .tensor(name)
            .unwrap_or_else(|_| panic!("missing tensor {name}"));
        assert_eq!(t.dtype(), safetensors::Dtype::F32);
        tensors_meta.push(serde_json::json!({ "name": name, "shape": t.shape() }));
        blob.extend_from_slice(t.data());
    }

    let header = serde_json::json!({
        "license": "OFL-1.1",
        "notice": "Derived from Gulzar (Copyright 2021 The Gulzar Project Authors, https://github.com/simoncozens/Gulzar), licensed under the SIL Open Font License 1.1.",
        "format": "neuraltype-field-v1",
        "script": "arabic",
        "style": style,
        "vocab": vocab,
        "arch": {
            "emb": emb_dim, "latent": latent, "c0": c0,
            "grid0": [
                (fmeta["h"].as_u64().unwrap() as usize + seed_div - 1) / seed_div,
                (fmeta["w"].as_u64().unwrap() as usize + seed_div - 1) / seed_div
            ],
            "chans": chans,
            "kernel": 4, "stride": 2, "padding": 1,
        },
        "canvas": {
            "w": fmeta["w"], "h": fmeta["h"],
            "origin_x": fmeta["origin_x"], "origin_y": fmeta["origin_y"],
            "em_px": fmeta["em_px"], "upm": fmeta["upm"],
            "spread_px": fmeta["spread_px"],
        },
        "tensors": tensors_meta,
    });
    let mut hjson = serde_json::to_vec(&header).unwrap();
    while (8 + hjson.len()) % 4 != 0 {
        hjson.push(b' ');
    }

    let mut f = std::fs::File::create(out_path).unwrap();
    f.write_all(b"NTF0").unwrap();
    f.write_all(&(hjson.len() as u32).to_le_bytes()).unwrap();
    f.write_all(&hjson).unwrap();
    f.write_all(&blob).unwrap();
    let total = 8 + hjson.len() + blob.len();
    println!(
        "wrote {out_path}: {total} bytes ({} weights, header {} bytes)",
        blob.len() / 4,
        hjson.len()
    );
}
