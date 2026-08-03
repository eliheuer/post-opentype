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

    // Fixed serialization order; names match the trainer's VarBuilder.
    let order = [
        "emb.weight",
        "l1.weight",
        "l1.bias",
        "l2.weight",
        "l2.bias",
        "disp.weight",
        "disp.bias",
        "d0.weight",
        "d0.bias",
        "d1.weight",
        "d1.bias",
        "d2.weight",
        "d2.bias",
        "d3.weight",
        "d3.bias",
        "d4.weight",
        "d4.bias",
    ];

    let mut tensors_meta = Vec::new();
    let mut blob: Vec<u8> = Vec::new();
    for name in order {
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
            "emb": 24, "latent": 256, "c0": 128,
            "grid0": [
                (fmeta["h"].as_u64().unwrap() as usize + 31) / 32,
                (fmeta["w"].as_u64().unwrap() as usize + 31) / 32
            ],
            "chans": [128, 64, 32, 16, 8, 1],
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
