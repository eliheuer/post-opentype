//! WASM bindings: load a .ntf neural font and shape/render text.

use neuraltype_core::shape::layout;
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
    /// Returns JSON: { width, grid_h, baseline, glyphs: [{ch, form, path}] }
    /// with paths as SVG strings in grid units (y-down).
    pub fn shape(&self, text: &str, elong: f64) -> String {
        let line = layout(&self.inner, text, elong.clamp(0.0, MAX_ELONG as f64));
        let glyphs: Vec<serde_json::Value> = line
            .glyphs
            .iter()
            .map(|g| {
                serde_json::json!({
                    "ch": g.ch.to_string(),
                    "form": format!("{:?}", g.form),
                    "path": g.path.to_svg(),
                })
            })
            .collect();
        serde_json::json!({
            "width": line.width,
            "grid_h": GRID_H,
            "baseline": BASELINE_ROW,
            "glyphs": glyphs,
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
