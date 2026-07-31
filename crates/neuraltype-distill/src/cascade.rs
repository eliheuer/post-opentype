//! Cascade figure data for the blog: shape one word with harfrust,
//! rasterize each cluster separately, and write an RGBA sheet with
//! clusters in alternating ink colors, the cluster origins joined in
//! red, and the baseline as a thin gray rule. Prints "W H" so the
//! designbot figure script can blit the blob.

use crate::fields::{parse_path, rasterize};

const EM_PX: f64 = 96.0;
const GREEN: [u8; 4] = [42, 163, 95, 255];
const GRAY: [u8; 4] = [110, 110, 110, 255];
const RED: [u8; 4] = [239, 68, 68, 255];
const RULE: [u8; 4] = [70, 70, 70, 255];

struct B(String);
impl ttf_parser::OutlineBuilder for B {
    fn move_to(&mut self, x: f32, y: f32) { self.0 += &format!("M{x} {y}"); }
    fn line_to(&mut self, x: f32, y: f32) { self.0 += &format!("L{x} {y}"); }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) { self.0 += &format!("Q{x1} {y1} {x} {y}"); }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) { self.0 += &format!("C{x1} {y1} {x2} {y2} {x} {y}"); }
    fn close(&mut self) { self.0.push('Z'); }
}

pub fn cascade(font_path: &str, word: &str, out_path: &str) {
    let font_bytes = std::fs::read(font_path).expect("source font not found");
    let font_ref = harfrust::FontRef::from_index(&font_bytes, 0).unwrap();
    let data = harfrust::ShaperData::new(&font_ref);
    let shaper = data.shaper(&font_ref).build();
    let ttf = ttf_parser::Face::parse(&font_bytes, 0).unwrap();
    let upm = ttf.units_per_em() as f64;

    let mut buf = harfrust::UnicodeBuffer::new();
    buf.push_str(word);
    buf.guess_segment_properties();
    let out = shaper.shape(buf, harfrust::ShapeOptions::default());

    // Group placed glyphs by source cluster; remember each cluster's
    // origin (the pen position where its first glyph lands).
    let mut clusters: Vec<(u32, Vec<(kurbo::BezPath, f64, f64)>, (f64, f64))> = Vec::new();
    let mut pen_x = 0i32;
    for (info, pos) in out.glyph_infos().iter().zip(out.glyph_positions()) {
        let mut b = B(String::new());
        ttf.outline_glyph(ttf_parser::GlyphId(info.glyph_id as u16), &mut b);
        let dx = (pen_x + pos.x_offset) as f64;
        let dy = pos.y_offset as f64;
        match clusters.last_mut() {
            Some((c, glyphs, _)) if *c == info.cluster => glyphs.push((parse_path(&b.0), dx, dy)),
            _ => clusters.push((info.cluster, vec![(parse_path(&b.0), dx, dy)], (dx, dy))),
        }
        pen_x += pos.x_advance;
    }

    // Global bounds (font units), baseline included.
    let mut bb = kurbo::Rect::new(0.0, 0.0, pen_x as f64, 1.0);
    for (_, glyphs, _) in &clusters {
        for (p, dx, dy) in glyphs {
            if p.elements().is_empty() { continue; }
            bb = bb.union(kurbo::Shape::bounding_box(p) + kurbo::Vec2::new(*dx, *dy));
        }
    }
    let scale = EM_PX / upm;
    let pad = 24.0 / scale;
    let (x0, y1) = (bb.x0 - pad, bb.y1 + pad);
    let w = ((bb.x1 - bb.x0 + 2.0 * pad) * scale).ceil() as usize;
    let h = ((bb.y1 - bb.y0 + 2.0 * pad) * scale).ceil() as usize;
    let to_px = |x: f64, y: f64| (((x - x0) * scale), ((y1 - y) * scale));

    let mut rgba = vec![0u8; w * h * 4];
    let mut put = |x: i64, y: i64, c: [u8; 4]| {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            let i = (y as usize * w + x as usize) * 4;
            rgba[i..i + 4].copy_from_slice(&c);
        }
    };

    // Baseline rule (y = 0 in font units).
    let (_, by) = to_px(0.0, 0.0);
    for x in 0..w as i64 {
        put(x, by as i64, RULE);
    }

    // Clusters in alternating inks.
    for (k, (_, glyphs, _)) in clusters.iter().enumerate() {
        let grid = rasterize(glyphs, w, h, scale, x0, y1);
        let ink = if k % 2 == 0 { GREEN } else { GRAY };
        for (i, &on) in grid.iter().enumerate() {
            if on {
                let idx = i * 4;
                // don't let the rule cut through letterforms
                let _ = idx;
                put((i % w) as i64, (i / w) as i64, ink);
            }
        }
    }

    // Origin chain in red: dots joined by a thin line.
    let pts: Vec<(f64, f64)> = clusters.iter().map(|(_, _, o)| to_px(o.0, o.1)).collect();
    for pair in pts.windows(2) {
        let (ax, ay) = pair[0];
        let (bx, by) = pair[1];
        let n = ((bx - ax).abs().max((by - ay).abs()).ceil() as usize).max(1);
        for s in 0..=n {
            let t = s as f64 / n as f64;
            put((ax + (bx - ax) * t) as i64, (ay + (by - ay) * t) as i64, RED);
        }
    }
    for (px, py) in &pts {
        for dy in -3i64..=3 {
            for dx in -3i64..=3 {
                if dx * dx + dy * dy <= 9 {
                    put(*px as i64 + dx, *py as i64 + dy, RED);
                }
            }
        }
    }

    std::fs::write(out_path, &rgba).unwrap();
    println!("{w} {h}");
}
