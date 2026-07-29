//! WASM bindings: load a .ntf neural font and shape/render text.

use neuraltype_core::shape::{layout, Dir};
use neuraltype_core::{font, model::NeuralFont, BASELINE_ROW, GRID_H, MAX_ELONG};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct NtfFont {
    inner: NeuralFont,
}

#[wasm_bindgen]
impl NtfFont {
    /// Parse a .ntf font (a serialized neural model) from bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<NtfFont, JsError> {
        font::load(bytes)
            .map(|inner| NtfFont { inner })
            .map_err(|e| JsError::new(&e))
    }

    /// Shape `text` at elongation `elong` ∈ [0, max_elong].
    /// Returns JSON: { width, grid_h, baseline, path, glyphs } where
    /// `path` is ONE SVG path for the whole line (connected letters
    /// are one continuous contour) in grid units (y-down), and
    /// `glyphs` is cluster metadata [{ch, form, x, advance}].
    pub fn shape(&self, text: &str, elong: f64, dir: &str) -> String {
        let dir = match dir {
            "rtl" => Dir::Rtl,
            "ltr" => Dir::Ltr,
            _ => Dir::Auto,
        };
        let line = layout(&self.inner, text, elong.clamp(0.0, MAX_ELONG as f64), dir);
        let glyphs: Vec<serde_json::Value> = line
            .glyphs
            .iter()
            .map(|g| {
                serde_json::json!({
                    "ch": g.ch.to_string(),
                    "form": format!("{:?}", g.form),
                    "x": g.x,
                    "advance": g.advance,
                })
            })
            .collect();
        let spans: Vec<serde_json::Value> = line
            .spans
            .iter()
            .map(|s| serde_json::json!({ "i": s.index, "x": s.x, "w": s.width }))
            .collect();
        serde_json::json!({
            "width": line.width,
            "grid_h": GRID_H,
            "baseline": BASELINE_ROW,
            "rtl": line.rtl,
            "path": line.path.to_svg(),
            "glyphs": glyphs,
            "spans": spans,
        })
        .to_string()
    }

    pub fn max_elong(&self) -> f64 {
        MAX_ELONG as f64
    }

    pub fn n_params(&self) -> usize {
        self.inner.mlp.n_params()
    }

    pub fn alphabet(&self) -> String {
        self.inner.alphabet.iter().collect()
    }
}
