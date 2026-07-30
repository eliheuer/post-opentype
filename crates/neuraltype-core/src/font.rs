//! The `.ntf` font file: a serialized neural font.
//!
//! Layout: magic "NTF0" ⧺ u32 LE header length ⧺ JSON header ⧺ raw
//! f32 LE weights. The header describes the alphabet and layer shapes;
//! the weights are the font. The header is padded with trailing spaces
//! (legal JSON whitespace) so the weight blob starts on a 4-byte
//! boundary and can be read in place from a memory-mapped file.

use crate::model::{Layer, Mlp, NeuralFont};
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 4] = b"NTF0";

#[derive(Serialize, Deserialize)]
pub struct Header {
    pub format: String,
    pub script: String,
    pub style: String,
    pub alphabet: String,
    /// Layer sizes, e.g. [63, 128, 128, 224].
    pub layers: Vec<usize>,
}

pub fn save(font: &NeuralFont, style: &str) -> Vec<u8> {
    let mut sizes = vec![font.mlp.layers[0].n_in];
    sizes.extend(font.mlp.layers.iter().map(|l| l.n_out));
    let header = Header {
        format: "neuraltype-mlp-v0".into(),
        script: "arabic".into(),
        style: style.into(),
        alphabet: font.alphabet.iter().collect(),
        layers: sizes,
    };
    let mut hjson = serde_json::to_vec(&header).unwrap();
    // Pad so the weight blob starts 4-byte aligned (trailing spaces
    // are legal JSON whitespace).
    while (8 + hjson.len()) % 4 != 0 {
        hjson.push(b' ');
    }
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(hjson.len() as u32).to_le_bytes());
    out.extend_from_slice(&hjson);
    for l in &font.mlp.layers {
        for v in l.w.iter().chain(&l.b) {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Parse just the JSON header of a .ntf file.
pub fn read_header(bytes: &[u8]) -> Result<(Header, usize), String> {
    if bytes.len() < 8 || &bytes[..4] != MAGIC {
        return Err("not a NeuralType (.ntf) file".into());
    }
    let hlen = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let header: Header =
        serde_json::from_slice(&bytes[8..8 + hlen]).map_err(|e| e.to_string())?;
    Ok((header, hlen))
}

pub fn load(bytes: &[u8]) -> Result<NeuralFont, String> {
    if bytes.len() < 8 || &bytes[..4] != MAGIC {
        return Err("not a NeuralType (.ntf) file".into());
    }
    let hlen = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let header: Header =
        serde_json::from_slice(&bytes[8..8 + hlen]).map_err(|e| e.to_string())?;
    let mut off = 8 + hlen;
    let mut read_f32s = |n: usize| -> Result<Vec<f32>, String> {
        let end = off + n * 4;
        if end > bytes.len() {
            return Err("truncated .ntf file".into());
        }
        let v = bytes[off..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        off = end;
        Ok(v)
    };
    let mut layers = Vec::new();
    for i in 0..header.layers.len() - 1 {
        let (n_in, n_out) = (header.layers[i], header.layers[i + 1]);
        let w = read_f32s(n_in * n_out)?;
        let b = read_f32s(n_out)?;
        let relu = i + 2 < header.layers.len(); // last layer linear
        layers.push(Layer { n_in, n_out, w, b, relu });
    }
    Ok(NeuralFont {
        mlp: Mlp { layers },
        alphabet: header.alphabet.chars().collect(),
    })
}
