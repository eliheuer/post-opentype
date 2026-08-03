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
    /// Per-caret-index placement overrides (field px, y-down): a
    /// dragged node edits the displacement chain, and everything
    /// after it follows. Cleared when the text changes.
    node_offsets: std::cell::RefCell<std::collections::HashMap<usize, (f64, f64)>>,
    /// img2bez-traced word outlines, keyed by word text (field px,
    /// y-down). Words being dragged bypass this and use the fast
    /// tracer until released.
    word_traces: std::cell::RefCell<std::collections::HashMap<String, kurbo::BezPath>>,
}

#[wasm_bindgen]
impl NtfFont {
    /// Parse a .ntf font (a serialized neural model) from bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<NtfFont, JsError> {
        console_error_panic_hook::set_once();
        if neuraltype_core::field_model::is_field_font(bytes) {
            return FieldFont::load(bytes)
                .map(|f| NtfFont {
                    inner: Font::Field(f),
                    node_offsets: Default::default(),
                    word_traces: Default::default(),
                })
                .map_err(|e| JsError::new(&e));
        }
        font::load(bytes)
            .map(|f| NtfFont {
                inner: Font::Kufic(f),
                node_offsets: Default::default(),
                word_traces: Default::default(),
            })
            .map_err(|e| JsError::new(&e))
    }

    /// Shape `text` at elongation `elong` ∈ [0, max_elong].
    /// Returns JSON: { width, grid_h, baseline, path, glyphs } where
    /// `path` is ONE SVG path for the whole line (connected letters
    /// are one continuous contour) in grid units (y-down), and
    /// `glyphs` is cluster metadata [{ch, form, x, advance}].
    /// Set the total placement offset for the cluster at caret index
    /// `i` (field px, y-down). Dragging a node calls this.
    pub fn set_node_offset(&self, i: usize, dx: f64, dy: f64) {
        self.node_offsets.borrow_mut().insert(i, (dx, dy));
    }

    pub fn clear_node_offsets(&self) {
        self.node_offsets.borrow_mut().clear();
    }

    pub fn shape(&self, text: &str, elong: f64, dir: &str) -> String {
        let inner = match &self.inner {
            Font::Field(f) => {
                return shape_field(f, text, &self.node_offsets.borrow(), &self.word_traces)
            }
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

    /// Selection cloud for field fonts: the union of the selected
    /// clusters' fields, traced at a raised iso level. The SDF gives
    /// the dilation for free, so the outline is a smooth blob that
    /// hugs the ink -- the cloud band of manuscript illumination,
    /// not a row of boxes. Returns an SVG path in line coordinates.
    pub fn selection_path(&self, text: &str, start: usize, end: usize) -> String {
        let f = match &self.inner {
            Font::Field(f) => f,
            _ => return String::new(),
        };
        if end <= start {
            return String::new();
        }
        let line = build_field_line(f, text, &self.node_offsets.borrow());
        let shift = line.width;
        let mut combined = String::new();
        for pw in &line.words {
            let a = start.max(pw.char_base);
            let b = end.min(pw.char_base + pw.n_chars);
            if a >= b {
                continue;
            }
            let mut ci = 0usize;
            let mask: Vec<bool> = pw
                .wf
                .clusters
                .iter()
                .map(|c| {
                    let nch = c.letters.chars().count();
                    let cs = pw.char_base + ci;
                    ci += nch;
                    cs < end && cs + nch > start
                })
                .collect();
            let sel =
                field_text::compose_clusters(f, pw.wf.clusters.clone(), Some(&mask));
            if sel.w == 0 {
                continue;
            }
            // +0.45 of the spread (8 px) dilates the zero contour by
            // about 3.6 px
            let dil: Vec<f32> = sel.grid.iter().map(|v| v + 0.45).collect();
            let path = field_text::trace_field_smooth(&dil, sel.w, sel.h);
            let path =
                kurbo::Affine::translate((sel.x0 + pw.dx + shift, sel.y0 - line.y_min)) * path;
            combined.push_str(&path.to_svg());
            combined.push(' ');
        }
        combined
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
/// works unchanged. Extra field-font data: `nodes`, one 2D point per
/// logical caret index, on the displacement chain.
struct PlacedWord {
    wf: field_text::WordField,
    dx: f64,
    char_base: usize,
    n_chars: usize,
    /// Ink edges in chain coordinates, measured before drag offsets.
    ink_l: f64,
    ink_r: f64,
    /// Chain-space y of the word's ink at its left (exit) edge: the
    /// end-of-word caret node sits here, on the tail of the last
    /// letter instead of floating mid-air.
    exit_y: f64,
    /// Chain-space y of the word's ink at its right (entry) edge:
    /// the before-the-word caret node sits here, on the first
    /// letter's entry stroke.
    entry_y: f64,
}

struct FieldLine {
    words: Vec<PlacedWord>,
    width: f64,
    y_min: f64,
    y_max: f64,
    space: f64,
    total_chars: usize,
}

type NodeOffsets = std::collections::HashMap<usize, (f64, f64)>;

fn build_field_line(f: &FieldFont, text: &str, offsets: &NodeOffsets) -> FieldLine {
    let em = f.canvas.em_px;
    // Gulzar's own space glyph advances 0.12 em (hmtx); match it.
    let space = 0.12 * em;
    let mut pen_right = 0.0f64;
    let mut y_min = -1.2 * em;
    let mut y_max = 0.5 * em;
    let mut words = Vec::new();
    let mut char_base = 0usize;
    for word in text.split(' ') {
        let n_chars = word.chars().count();
        if word.is_empty() {
            pen_right -= space;
            char_base += 1;
            continue;
        }
        // layout, then apply dragged-node offsets cumulatively: an
        // offset at cluster k moves k and every cluster after it,
        // because the chain is relative. Placement anchors on the
        // UNOFFSET ink, so a drag moves letters on screen instead of
        // being cancelled by the pen re-anchoring.
        let clusters = field_text::layout_word(f, word);
        let mut ci = 0usize;
        let mut has_off = false;
        for c in &clusters {
            if offsets.contains_key(&(char_base + ci)) {
                has_off = true;
            }
            ci += c.letters.chars().count();
        }
        let base_wf = field_text::compose_clusters(f, clusters.clone(), None);
        if base_wf.w == 0 {
            char_base += n_chars + 1;
            continue;
        }
        let ink = |wf: &field_text::WordField| {
            let (mut ix0, mut ix1, mut iy0, mut iy1) =
                (usize::MAX, 0usize, usize::MAX, 0usize);
            for y in 0..wf.h {
                for x in 0..wf.w {
                    if wf.grid[y * wf.w + x] >= 0.0 {
                        ix0 = ix0.min(x);
                        ix1 = ix1.max(x);
                        iy0 = iy0.min(y);
                        iy1 = iy1.max(y);
                    }
                }
            }
            (ix0, ix1, iy0, iy1)
        };
        let (bx0, bx1, _, _) = ink(&base_wf);
        if bx0 == usize::MAX {
            char_base += n_chars + 1;
            continue;
        }
        // ink edges in chain coordinates, from the unoffset word
        let ink_r = base_wf.x0 + bx1 as f64 + 1.0;
        let ink_l = base_wf.x0 + bx0 as f64;
        // ink exit height: centroid of the leftmost few ink columns
        let mut ysum = 0.0f64;
        let mut yn = 0usize;
        for y in 0..base_wf.h {
            for x in bx0..(bx0 + 4).min(base_wf.w) {
                if base_wf.grid[y * base_wf.w + x] >= 0.0 {
                    ysum += y as f64;
                    yn += 1;
                }
            }
        }
        let exit_y = base_wf.y0 + if yn > 0 { ysum / yn as f64 } else { 0.0 };
        let mut ysum_r = 0.0f64;
        let mut yn_r = 0usize;
        for y in 0..base_wf.h {
            for x in bx1.saturating_sub(3)..=bx1 {
                if base_wf.grid[y * base_wf.w + x] >= 0.0 {
                    ysum_r += y as f64;
                    yn_r += 1;
                }
            }
        }
        let entry_y = base_wf.y0 + if yn_r > 0 { ysum_r / yn_r as f64 } else { 0.0 };
        let wf = if has_off {
            let mut moved = clusters;
            let mut cum = (0.0f64, 0.0f64);
            let mut ci = 0usize;
            for c in moved.iter_mut() {
                if let Some(&(dx, dy)) = offsets.get(&(char_base + ci)) {
                    cum.0 += dx;
                    cum.1 += dy;
                }
                c.ox += cum.0;
                c.oy += cum.1;
                ci += c.letters.chars().count();
            }
            field_text::compose_clusters(f, moved, None)
        } else {
            base_wf
        };
        if wf.w == 0 {
            char_base += n_chars + 1;
            continue;
        }
        let (_, _, ry0, ry1) = ink(&wf);
        // place word: right edge of its unoffset INK at pen_right
        let dx = pen_right - ink_r;
        if ry0 != usize::MAX {
            y_min = y_min.min(wf.y0 + ry0 as f64);
            y_max = y_max.max(wf.y0 + ry1 as f64 + 1.0);
        }
        pen_right -= (ink_r - ink_l) + space;
        words.push(PlacedWord {
            wf,
            dx,
            char_base,
            n_chars,
            ink_l,
            ink_r,
            exit_y,
            entry_y,
        });
        char_base += n_chars + 1;
    }
    let width = -pen_right - space.min(-pen_right);
    FieldLine {
        words,
        width,
        y_min,
        y_max,
        space,
        total_chars: text.chars().count(),
    }
}

/// Trace a composed word with img2bez's type-quality fitter: the SDF
/// upsampled with a true anti-aliased coverage ramp, traced, and
/// mapped back to field pixels (y-down). None on any failure -- the
/// caller falls back to the fast tracer.
fn trace_word_i2b(f: &FieldFont, wf: &field_text::WordField) -> Option<kurbo::BezPath> {
    if wf.w == 0 || wf.h == 0 {
        return None;
    }
    let spread = f.canvas.spread_px;
    let s = ((768.0 / wf.h as f64).ceil() as usize).clamp(3, 10);
    let (uw, uh) = (wf.w * s, wf.h * s);
    let mut gray = vec![0u8; uw * uh];
    for y in 0..uh {
        let fy = ((y as f32 + 0.5) / s as f32 - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(wf.h - 1);
        let y1 = (y0 + 1).min(wf.h - 1);
        let ty = (fy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..uw {
            let fx = ((x as f32 + 0.5) / s as f32 - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(wf.w - 1);
            let x1 = (x0 + 1).min(wf.w - 1);
            let tx = (fx - x0 as f32).clamp(0.0, 1.0);
            let a = wf.grid[y0 * wf.w + x0] * (1.0 - tx) + wf.grid[y0 * wf.w + x1] * tx;
            let b = wf.grid[y1 * wf.w + x0] * (1.0 - tx) + wf.grid[y1 * wf.w + x1] * tx;
            let v = (a * (1.0 - ty) + b * ty) as f64;
            let d = v * spread * s as f64;
            gray[y * uw + x] = (((d / 1.2) + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    let mut png_bytes: Vec<u8> = Vec::new();
    image::GrayImage::from_raw(uw as u32, uh as u32, gray)?
        .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
        .ok()?;
    let mut opts = img2bez::TraceOptions::default();
    opts.rtl_start = true;
    // all-curves smooth mode: tangent-continuous outlines, no hard
    // corners -- the right register for nastaliq
    opts.mode = img2bez::TraceMode::Smooth;
    let outline = img2bez::trace(&png_bytes, &opts).ok()?;
    let path = kurbo::BezPath::from_svg(&outline.to_svg_path()).ok()?;
    // unit space (y-up, image height = em_height) -> field px (y-down)
    let k = uh as f64 / (opts.em_height * s as f64);
    let a = kurbo::Affine::new([k, 0.0, 0.0, -k, 0.0, wf.h as f64]);
    Some(a * path)
}

fn shape_field(
    f: &FieldFont,
    text: &str,
    offsets: &NodeOffsets,
    traces: &std::cell::RefCell<std::collections::HashMap<String, kurbo::BezPath>>,
) -> String {
    let line = build_field_line(f, text, offsets);
    let shift = line.width;
    let y_min = line.y_min;
    let baseline_y = -y_min;
    let mut combined = String::new();
    let mut spans: Vec<serde_json::Value> = Vec::new();
    let n = line.total_chars;
    let mut nodes: Vec<(f64, f64)> = vec![(f64::NAN, f64::NAN); n + 1];

    let words_text: Vec<&str> = text.split(' ').collect();
    let _ = &words_text;
    for pw in &line.words {
        let wf = &pw.wf;
        let word_str: String = text
            .chars()
            .skip(pw.char_base)
            .take(pw.n_chars)
            .collect();
        let word_has_offsets = (pw.char_base..pw.char_base + pw.n_chars)
            .any(|i| offsets.contains_key(&i));
        let cached = traces.borrow().get(&word_str).cloned();
        let path = if word_has_offsets {
            // live drag: fast tracer, no cache
            field_text::trace_field_smooth(&wf.grid, wf.w, wf.h)
        } else if let Some(p) = cached {
            p
        } else {
            let p = trace_word_i2b(f, wf)
                .unwrap_or_else(|| field_text::trace_field_smooth(&wf.grid, wf.w, wf.h));
            traces.borrow_mut().insert(word_str.clone(), p.clone());
            p
        };
        let path = kurbo::Affine::translate((wf.x0 + pw.dx + shift, wf.y0 - y_min)) * path;
        combined.push_str(&path.to_svg());
        combined.push(' ');

        // visual join point between two clusters: the deepest cell of
        // their fields' intersection -- the middle of the connecting
        // stroke, which is where an insertion visually breaks the
        // word. None when the letters do not touch.
        let join_point = |ka: &field_text::Cluster,
                          kb: &field_text::Cluster|
         -> Option<(f64, f64)> {
            let (cw_i, ch_i) = (f.canvas.w as i64, f.canvas.h as i64);
            let (cox, coy) = (f.canvas.origin_x, f.canvas.origin_y);
            let ga = f.glyph(ka.feats);
            let gb = f.glyph(kb.feats);
            let ax0 = (ka.ox - cox).round() as i64;
            let ay0 = (ka.oy - coy).round() as i64;
            let bx0 = (kb.ox - cox).round() as i64;
            let by0 = (kb.oy - coy).round() as i64;
            let x0 = ax0.max(bx0);
            let y0 = ay0.max(by0);
            let x1 = (ax0 + cw_i).min(bx0 + cw_i);
            let y1 = (ay0 + ch_i).min(by0 + ch_i);
            let mut best = f32::MIN;
            let mut bp = None;
            for y in y0..y1 {
                for x in x0..x1 {
                    let va = ga.field[((y - ay0) * cw_i + (x - ax0)) as usize];
                    let vb = gb.field[((y - by0) * cw_i + (x - bx0)) as usize];
                    let m = va.min(vb);
                    if m > best {
                        best = m;
                        bp = Some((x as f64 + 0.5, y as f64 + 0.5));
                    }
                }
            }
            if best >= 0.0 {
                bp
            } else {
                None
            }
        };

        let pen_right_word = pw.dx + pw.ink_r + shift;
        let left_ink = pw.ink_l + pw.dx + shift;
        let cl = &wf.clusters;
        let mut ci = 0usize;
        for (k, c) in cl.iter().enumerate() {
            let right = if k == 0 { pen_right_word } else { cl[k - 1].ox + pw.dx + shift };
            let left = if k + 1 < cl.len() { cl[k + 1].ox + pw.dx + shift } else { left_ink };
            let nch = c.letters.chars().count();
            let cw_ = (right - left).max(1.0) / nch as f64;
            let ox = c.ox + pw.dx + shift;
            let oy = c.oy - y_min;
            for j in 0..nch {
                let i = pw.char_base + ci + j;
                spans.push(serde_json::json!({
                    "i": i,
                    "x": right - (j as f64 + 1.0) * cw_,
                    "w": cw_,
                }));
                if i <= n {
                    if k == 0 && j == 0 {
                        // the before-the-word slot: on the first
                        // letter's ink entry, not the abstract pen
                        // origin
                        nodes[i] =
                            (pw.ink_r + pw.dx + shift + line.space * 0.25, pw.entry_y - y_min);
                    } else if nch == 1 {
                        // interior break: the visual junction where
                        // this letter's ink meets the previous
                        // letter's ink; chain origin when they do
                        // not touch
                        nodes[i] = match join_point(&cl[k - 1], c) {
                            Some((jx, jy)) => (jx + pw.dx + shift, jy - y_min),
                            None => (ox, oy),
                        };
                    } else {
                        // ligatures: one node per character cell,
                        // spread across the cluster's visual span --
                        // the origin can sit at either end of a wide
                        // ligature, and stacking nodes there clumps
                        // the strand
                        nodes[i] = (right - (j as f64 + 0.5) * cw_, oy);
                    }
                }
            }
            ci += nch;
        }
        // end-of-word caret: just past the left ink edge, at the
        // height where the word's ink actually exits
        let end_i = pw.char_base + pw.n_chars;
        if end_i <= n {
            nodes[end_i] = (left_ink - line.space * 0.5, pw.exit_y - y_min);
        }
    }
    // fill gaps (leading/trailing spaces, unrendered words):
    // forward then backward copy of the nearest known node
    let mut last: Option<(f64, f64)> = None;
    for p in nodes.iter_mut() {
        if p.0.is_nan() {
            if let Some(q) = last {
                *p = q;
            }
        } else {
            last = Some(*p);
        }
    }
    let mut next: Option<(f64, f64)> = None;
    for p in nodes.iter_mut().rev() {
        if p.0.is_nan() {
            *p = next.unwrap_or((shift, baseline_y));
        } else {
            next = Some(*p);
        }
    }
    let nodes_json: Vec<serde_json::Value> = nodes
        .iter()
        .enumerate()
        .map(|(i, p)| serde_json::json!({ "i": i, "x": p.0, "y": p.1 }))
        .collect();

    serde_json::json!({
        "width": line.width,
        "grid_h": (line.y_max - line.y_min).ceil(),
        "baseline": baseline_y.round(),
        "rtl": true,
        "path": combined,
        "glyphs": [],
        "spans": spans,
        "nodes": nodes_json,
        "field": true,
        "em_px": f.canvas.em_px,
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
