//! WASM bindings: load a .ntf neural font and shape/render text.

use neuraltype_core::field_model::FieldFont;
use neuraltype_core::field_text;
use neuraltype_core::shape::{layout, Dir};
use neuraltype_core::{font, model::NeuralFont, BASELINE_ROW, GRID_H, MAX_ELONG};
use wasm_bindgen::prelude::*;

enum Font {
    Kufic(NeuralFont),
    Field(FieldFont),
}

#[wasm_bindgen]
pub struct NtfFont {
    inner: Font,
}

#[wasm_bindgen]
impl NtfFont {
    /// Parse a .ntf font (a serialized neural model) from bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<NtfFont, JsError> {
        if neuraltype_core::field_model::is_field_font(bytes) {
            return FieldFont::load(bytes)
                .map(|f| NtfFont { inner: Font::Field(f) })
                .map_err(|e| JsError::new(&e));
        }
        font::load(bytes)
            .map(|f| NtfFont { inner: Font::Kufic(f) })
            .map_err(|e| JsError::new(&e))
    }

    /// Shape `text` at elongation `elong` ∈ [0, max_elong].
    /// Returns JSON: { width, grid_h, baseline, path, glyphs } where
    /// `path` is ONE SVG path for the whole line (connected letters
    /// are one continuous contour) in grid units (y-down), and
    /// `glyphs` is cluster metadata [{ch, form, x, advance}].
    pub fn shape(&self, text: &str, elong: f64, dir: &str) -> String {
        let inner = match &self.inner {
            Font::Field(f) => return shape_field(f, text),
            Font::Kufic(f) => f,
        };
        let dir = match dir {
            "rtl" => Dir::Rtl,
            "ltr" => Dir::Ltr,
            _ => Dir::Auto,
        };
        let line = layout(inner, text, elong.clamp(0.0, MAX_ELONG as f64), dir);
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
        match &self.inner {
            Font::Kufic(f) => f.mlp.n_params(),
            Font::Field(f) => f.n_params(),
        }
    }

    pub fn alphabet(&self) -> String {
        match &self.inner {
            Font::Kufic(f) => f.alphabet.iter().collect(),
            Font::Field(f) => f.alphabet(),
        }
    }
}

/// Layout for field fonts: words composed by the model, laid out RTL
/// on a shared baseline. Coordinates are in field pixels (em_px per
/// em); the JSON contract matches the v0 shape() so the demo island
/// works unchanged.
fn shape_field(f: &FieldFont, text: &str) -> String {
    let em = f.canvas.em_px;
    let space = 0.4 * em;
    let mut pen_right = 0.0f64;
    let mut paths: Vec<String> = Vec::new();
    let mut spans: Vec<serde_json::Value> = Vec::new();
    let mut y_min = -1.2 * em;
    let mut y_max = 0.5 * em;

    // logical char index base per word
    let mut char_base = 0usize;
    for word in text.split(' ') {
        let n_chars = word.chars().count();
        if word.is_empty() {
            // an explicit space character: give it a span
            spans.push(serde_json::json!({
                "i": char_base, "x": pen_right - space, "w": space }));
            pen_right -= space;
            char_base += 1;
            continue;
        }
        let wf = field_text::compose_word(f, word);
        if wf.w == 0 {
            char_base += n_chars + 1;
            continue;
        }
        // place word: right edge of its grid at pen_right
        let dx = pen_right - (wf.x0 + wf.w as f64);
        let path = field_text::trace_field(&wf.grid, wf.w, wf.h);
        let path = kurbo::Affine::translate((wf.x0 + dx, wf.y0)) * path;
        paths.push(path.to_svg());
        y_min = y_min.min(wf.y0);
        y_max = y_max.max(wf.y0 + wf.h as f64);

        // spans: cluster k covers x from its origin to the previous
        // cluster's origin (RTL); first cluster reaches the right edge.
        let cl = &wf.clusters;
        let mut ci = 0usize; // char offset within word
        for (k, c) in cl.iter().enumerate() {
            let right = if k == 0 { pen_right } else { cl[k - 1].ox + dx };
            let left = if k + 1 < cl.len() { cl[k + 1].ox + dx } else { wf.x0 + dx };
            let nch = c.letters.chars().count();
            // one span per source char, splitting the cluster width
            let cw = (right - left).max(1.0) / nch as f64;
            for j in 0..nch {
                spans.push(serde_json::json!({
                    "i": char_base + ci + j,
                    "x": right - (j as f64 + 1.0) * cw,
                    "w": cw,
                }));
            }
            let _ = left;
            ci += nch;
        }
        pen_right -= wf.w as f64 + space;
        char_base += n_chars + 1; // + the following space
    }
    let width = -pen_right - space.min(-pen_right);
    // shift everything right so the line starts at x = 0
    let shift = width;
    let paths: Vec<String> = paths
        .iter()
        .map(|p| p.clone())
        .collect();
    // note: paths are in pen coordinates (negative x); the client
    // receives a combined path already shifted.
    let mut combined = String::new();
    for p in &paths {
        combined.push_str(p);
        combined.push(' ');
    }
    let spans: Vec<serde_json::Value> = spans
        .iter()
        .map(|s| {
            serde_json::json!({
                "i": s["i"],
                "x": s["x"].as_f64().unwrap() + shift,
                "w": s["w"],
            })
        })
        .collect();
    serde_json::json!({
        "width": width,
        "grid_h": (y_max - y_min).ceil(),
        "baseline": (-y_min).round(),
        "rtl": true,
        "path": translate_svg(&combined, shift, -y_min),
        "glyphs": [],
        "spans": spans,
        "field": true,
        "em_px": em,
    })
    .to_string()
}

/// Translate an SVG path string by (dx, dy) by reparsing through kurbo.
fn translate_svg(svg: &str, dx: f64, dy: f64) -> String {
    match kurbo::BezPath::from_svg(svg) {
        Ok(p) => (kurbo::Affine::translate((dx, dy)) * p).to_svg(),
        Err(_) => svg.to_string(),
    }
}
