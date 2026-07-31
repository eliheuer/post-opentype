//! Compare a trained field .ntf against the source font, word by
//! word: ground truth (harfrust shaping + outline rasterization) on
//! top, the model's traced field below. Output is a PGM sheet.

use crate::fields::{parse_path, rasterize};
use neuraltype_core::{field_model::FieldFont, field_text};
use std::io::Write as _;

const EM_PX: f64 = 64.0;

/// Rasterize the ground truth for one word at EM_PX, baseline-relative.
fn ground_truth(font_bytes: &[u8], word: &str) -> (Vec<bool>, usize, usize) {
    let font_ref = harfrust::FontRef::from_index(font_bytes, 0).unwrap();
    let data = harfrust::ShaperData::new(&font_ref);
    let shaper = data.shaper(&font_ref).build();
    let ttf = ttf_parser::Face::parse(font_bytes, 0).unwrap();
    let upm = ttf.units_per_em() as f64;

    let mut buf = harfrust::UnicodeBuffer::new();
    buf.push_str(word);
    buf.guess_segment_properties();
    let out = shaper.shape(buf, harfrust::ShapeOptions::default());

    struct B(String);
    impl ttf_parser::OutlineBuilder for B {
        fn move_to(&mut self, x: f32, y: f32) { self.0 += &format!("M{x} {y}"); }
        fn line_to(&mut self, x: f32, y: f32) { self.0 += &format!("L{x} {y}"); }
        fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) { self.0 += &format!("Q{x1} {y1} {x} {y}"); }
        fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) { self.0 += &format!("C{x1} {y1} {x2} {y2} {x} {y}"); }
        fn close(&mut self) { self.0.push('Z'); }
    }

    let mut placed: Vec<(kurbo::BezPath, f64, f64)> = Vec::new();
    let mut pen_x = 0i32;
    for (info, pos) in out.glyph_infos().iter().zip(out.glyph_positions()) {
        let mut b = B(String::new());
        ttf.outline_glyph(ttf_parser::GlyphId(info.glyph_id as u16), &mut b);
        placed.push((
            parse_path(&b.0),
            (pen_x + pos.x_offset) as f64,
            pos.y_offset as f64,
        ));
        pen_x += pos.x_advance;
    }
    // bounds in font units
    let mut bb: Option<kurbo::Rect> = None;
    for (p, dx, dy) in &placed {
        if p.elements().is_empty() { continue; }
        let r = kurbo::Shape::bounding_box(p) + kurbo::Vec2::new(*dx, *dy);
        bb = Some(bb.map_or(r, |b| b.union(r)));
    }
    let bb = bb.unwrap_or(kurbo::Rect::new(0.0, 0.0, 1.0, 1.0));
    let scale = EM_PX / upm;
    let pad = 4.0 / scale;
    let (x0, y0, x1, y1) = (bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad);
    let w = ((x1 - x0) * scale).ceil() as usize;
    let h = ((y1 - y0) * scale).ceil() as usize;
    let grid = rasterize(&placed, w, h, scale, x0, y1);
    (grid, w, h)
}

pub fn compare(font_path: &str, ntf_path: &str, out_path: &str, words: &[String]) {
    let font_bytes = std::fs::read(font_path).expect("source font not found");
    let ntf_bytes = std::fs::read(ntf_path).expect(".ntf not found");
    let field_font = FieldFont::load(&ntf_bytes).expect("bad field font");

    // Render each word both ways into (grid, w, h) panels.
    let mut panels: Vec<(Vec<u8>, usize, usize)> = Vec::new();
    for word in words {
        let (gt, gw, gh) = ground_truth(&font_bytes, word);
        panels.push((gt.iter().map(|&b| if b { 255u8 } else { 0 }).collect(), gw, gh));
        let wf = field_text::compose_word(&field_font, word);
        let model: Vec<u8> = wf.grid.iter().map(|&v| if v >= 0.0 { 255u8 } else { 0 }).collect();
        panels.push((model, wf.w, wf.h));
    }

    // Sheet: 2 rows per word pair? Lay out as columns of (gt, model).
    let gap = 8usize;
    let col_w: Vec<usize> = (0..words.len())
        .map(|i| panels[2 * i].1.max(panels[2 * i + 1].1))
        .collect();
    let row0_h = (0..words.len()).map(|i| panels[2 * i].2).max().unwrap();
    let row1_h = (0..words.len()).map(|i| panels[2 * i + 1].2).max().unwrap();
    let sheet_w: usize = col_w.iter().sum::<usize>() + gap * (words.len() + 1);
    let sheet_h = row0_h + row1_h + gap * 3;
    let mut sheet = vec![32u8; sheet_w * sheet_h];
    let mut x_off = gap;
    for i in 0..words.len() {
        for (row, y_off) in [(2 * i, gap), (2 * i + 1, gap * 2 + row0_h)] {
            let (px, pw, ph) = &panels[row];
            for y in 0..*ph {
                for x in 0..*pw {
                    sheet[(y_off + y) * sheet_w + x_off + x] = px[y * pw + x];
                }
            }
        }
        x_off += col_w[i] + gap;
    }
    if out_path.ends_with(".rgba") {
        // Blog-figure mode: transparent background, teacher row in
        // the outline gray, model row in the ink green.
        let split = row0_h + gap * 2;
        let mut rgba = vec![0u8; sheet_w * sheet_h * 4];
        for (i, &v) in sheet.iter().enumerate() {
            if v > 128 {
                let c: [u8; 4] =
                    if i / sheet_w < split { [110, 110, 110, 255] } else { [42, 163, 95, 255] };
                rgba[i * 4..i * 4 + 4].copy_from_slice(&c);
            }
        }
        std::fs::write(out_path, &rgba).unwrap();
        println!("{sheet_w} {sheet_h}");
        return;
    }
    let mut f = std::fs::File::create(out_path).unwrap();
    writeln!(f, "P5\n{sheet_w} {sheet_h}\n255").unwrap();
    f.write_all(&sheet).unwrap();
    println!("wrote {out_path}: top = Gulzar ground truth, bottom = model");
}
