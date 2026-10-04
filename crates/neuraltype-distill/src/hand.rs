//! Hand: turn labeled phrases into training rows.
//!
//! The distilled dataset comes from a teacher font. This module adds
//! rows that come from a designer instead: a phrase of calligraphy
//! whose ink is labeled letter by letter. The rows use the same
//! record format and the same canvas as the teacher's, so they
//! append to a teacher dataset and the trainer fine-tunes on both.
//!
//! Input is a neural source (a `.nufo` directory): every labeled
//! sample of every canvas becomes one phrase. A phrase file (JSON),
//! written by `distill standin`, is accepted too, for tests:
//!
//! ```json
//! { "text": "بسم الله", "upm": 1000,
//!   "outline": "M.. Z",
//!   "clusters": [
//!     { "letters": "ب", "origin": [3610, 410],
//!       "regions": [[[x, y], ...]], "paths": ["M.. Z"] } ] }
//! ```
//!
//! `origin` is optional: without it the cluster gets the origin that
//! centers its ink in the canvas, which is also the origin that lets
//! the largest cluster fit.
//!
//! Coordinates are y-up, in units of `upm` per em. `clusters` lists
//! the letters of every word in logical order, spaces left out. A
//! cluster's ink is the phrase `outline` inside its `regions`
//! (polygons, which may overlap those of its neighbors), plus its own
//! `paths` (whole contours it owns, or a complete outline when the
//! phrase has no shared `outline`).
//!
//! Two things differ from teacher rows. Labeled rows carry `pri: 1`,
//! so they replace teacher rows with the same context. And each space
//! inside a phrase becomes a row of its own, letters " ", whose
//! displacement runs from the last origin of one word to the first
//! origin of the next: the phrase keeps its composition.

use crate::fields::{rasterize, sdf_from_grid};
use neuraltype_core::field_text::cluster_ranges;
use serde::{Deserialize, Serialize};
use std::io::Write as _;

#[derive(Serialize, Deserialize, Clone)]
pub struct PhraseCluster {
    pub letters: String,
    /// Where the cluster's origin sits. When absent, the origin that
    /// centers the cluster's ink in the canvas is used.
    #[serde(default)]
    pub origin: Option<[f64; 2]>,
    #[serde(default)]
    pub regions: Vec<Vec<[f64; 2]>>,
    #[serde(default)]
    pub paths: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Phrase {
    pub text: String,
    pub upm: f64,
    #[serde(default)]
    pub outline: Option<String>,
    /// The line the phrase sits on, in phrase coordinates.
    #[serde(default)]
    pub baseline_y: f64,
    pub clusters: Vec<PhraseCluster>,
}

struct Canvas {
    w: usize,
    h: usize,
    origin_x: f64,
    origin_y: f64,
    px_per_unit: f64,
    spread_px: f64,
}

const SS: usize = 4;

/// Rasterize one cluster's ink on the shared canvas (supersampled),
/// with the cluster origin at the canvas anchor.
fn cluster_ink(
    c: &PhraseCluster,
    origin: (f64, f64),
    outline: Option<&kurbo::BezPath>,
    k: f64,
    cv: &Canvas,
) -> Vec<bool> {
    let (sw, sh) = (cv.w * SS, cv.h * SS);
    let scale = cv.px_per_unit * SS as f64;
    // font-unit position of pixel column 0 and of the top row
    let x0 = origin.0 - cv.origin_x / cv.px_per_unit;
    let y_top = origin.1 + cv.origin_y / cv.px_per_unit;
    ink_grid(c, outline, k, sw, sh, scale, x0, y_top)
}

/// A cluster's ink on any grid: `scale` px per font unit, pixel
/// column 0 at font-unit `x0`, top row at font-unit `y_top`.
#[allow(clippy::too_many_arguments)]
fn ink_grid(
    c: &PhraseCluster,
    outline: Option<&kurbo::BezPath>,
    k: f64,
    sw: usize,
    sh: usize,
    scale: f64,
    x0: f64,
    y_top: f64,
) -> Vec<bool> {
    let to_units = kurbo::Affine::scale(k);
    let mut ink = vec![false; sw * sh];
    if let (Some(outline), false) = (outline, c.regions.is_empty()) {
        let whole = rasterize(&[(to_units * outline.clone(), 0.0, 0.0)], sw, sh, scale, x0, y_top);
        let mut region = kurbo::BezPath::new();
        for poly in &c.regions {
            for (i, p) in poly.iter().enumerate() {
                if i == 0 {
                    region.move_to((p[0], p[1]));
                } else {
                    region.line_to((p[0], p[1]));
                }
            }
            region.close_path();
        }
        let mask = rasterize(&[(to_units * region, 0.0, 0.0)], sw, sh, scale, x0, y_top);
        for i in 0..ink.len() {
            ink[i] = whole[i] && mask[i];
        }
    }
    if !c.paths.is_empty() {
        let own: Vec<(kurbo::BezPath, f64, f64)> = c
            .paths
            .iter()
            .map(|d| (to_units * kurbo::BezPath::from_svg(d).expect("cluster path"), 0.0, 0.0))
            .collect();
        let grid = rasterize(&own, sw, sh, scale, x0, y_top);
        for i in 0..ink.len() {
            ink[i] |= grid[i];
        }
    }
    ink
}

/// The box of a cluster's ink in font units, measured at `scale` px
/// per unit. None when the cluster has no ink.
fn ink_box(
    c: &PhraseCluster,
    outline: Option<&kurbo::BezPath>,
    k: f64,
    scale: f64,
) -> Option<kurbo::Rect> {
    // Bound the search by the geometry, then measure the real ink.
    let mut bb: Option<kurbo::Rect> = None;
    let mut grow = |r: kurbo::Rect| bb = Some(bb.map_or(r, |b| b.union(r)));
    for poly in &c.regions {
        for p in poly {
            grow(kurbo::Rect::new(p[0] * k, p[1] * k, p[0] * k, p[1] * k));
        }
    }
    for d in &c.paths {
        let p = kurbo::Affine::scale(k) * kurbo::BezPath::from_svg(d).expect("cluster path");
        if !p.elements().is_empty() {
            grow(kurbo::Shape::bounding_box(&p));
        }
    }
    let bb = bb?;
    let (w, h) = ((bb.width() * scale).ceil() as usize + 2, (bb.height() * scale).ceil() as usize + 2);
    let (x0, y_top) = (bb.x0 - 1.0 / scale, bb.y1 + 1.0 / scale);
    let ink = ink_grid(c, outline, k, w, h, scale, x0, y_top);
    let (mut c0, mut c1, mut r0, mut r1) = (usize::MAX, 0usize, usize::MAX, 0usize);
    for (i, _) in ink.iter().enumerate().filter(|(_, &v)| v) {
        c0 = c0.min(i % w);
        c1 = c1.max(i % w);
        r0 = r0.min(i / w);
        r1 = r1.max(i / w);
    }
    if c0 == usize::MAX {
        return None;
    }
    Some(kurbo::Rect::new(
        x0 + c0 as f64 / scale,
        y_top - (r1 + 1) as f64 / scale,
        x0 + (c1 + 1) as f64 / scale,
        y_top - r0 as f64 / scale,
    ))
}

/// The origin (font units) that centers a cluster's ink in the canvas.
fn centered_origin(
    c: &PhraseCluster,
    outline: Option<&kurbo::BezPath>,
    k: f64,
    cv: &Canvas,
) -> Option<(f64, f64)> {
    let ink = ink_box(c, outline, k, cv.px_per_unit)?;
    let (cx, cy) = (ink.center().x, ink.center().y);
    Some((
        cx - (cv.w as f64 / 2.0 - cv.origin_x) / cv.px_per_unit,
        cy - (cv.origin_y - cv.h as f64 / 2.0) / cv.px_per_unit,
    ))
}

/// A phrase's words, and its clusters grouped into the engine's
/// clusters word by word: the engine fuses some letters (لا, لله in
/// الله).
fn lay_out(path: &str, phrase: &Phrase) -> (Vec<Vec<char>>, Vec<Vec<PhraseCluster>>) {
    let words: Vec<Vec<char>> = phrase
        .text
        .split(' ')
        .filter(|w| !w.is_empty())
        .map(|w| w.chars().collect())
        .collect();
    let mut input = phrase.clusters.iter();
    let mut laid: Vec<Vec<PhraseCluster>> = Vec::new();
    for chars in &words {
        let mut word_clusters = Vec::new();
        for (a, b) in cluster_ranges(chars) {
            let want: String = chars[a..b].iter().collect();
            let mut merged: Option<PhraseCluster> = None;
            while merged.as_ref().map_or(true, |m| m.letters != want) {
                let c = input.next().unwrap_or_else(|| {
                    panic!("{path}: ran out of clusters at {want:?} in {:?}", phrase.text)
                });
                match merged.as_mut() {
                    None => merged = Some(c.clone()),
                    Some(m) => {
                        // a fused cluster keeps the first part's
                        // origin only if every part had one
                        if c.origin.is_none() {
                            m.origin = None;
                        }
                        m.letters.push_str(&c.letters);
                        m.regions.extend(c.regions.iter().cloned());
                        m.paths.extend(c.paths.iter().cloned());
                    }
                }
                let got = &merged.as_ref().unwrap().letters;
                assert!(
                    want.starts_with(got.as_str()),
                    "{path}: clusters do not match the text: got {got:?}, want {want:?}"
                );
            }
            word_clusters.push(merged.unwrap());
        }
        laid.push(word_clusters);
    }
    assert!(input.next().is_none(), "{path}: more clusters than letters in the text");
    (words, laid)
}

/// Whether any ink touches the canvas edge (the cluster is clipped).
fn clipped(ink: &[bool], sw: usize, sh: usize) -> bool {
    (0..sw).any(|x| ink[x] || ink[(sh - 1) * sw + x])
        || (0..sh).any(|y| ink[y * sw] || ink[y * sw + sw - 1])
}

/// Every labeled sample of a neural source, as phrases named
/// `canvas #n`. A sample that is not ready stops the run with the
/// reason: training must not quietly skip work.
fn nufo_phrases(path: &str) -> Vec<(String, Phrase)> {
    let source = nufo::Source::load(std::path::Path::new(path)).unwrap_or_else(|e| panic!("{e}"));
    let mut out = Vec::new();
    for canvas in &source.canvases {
        for (n, sample) in canvas.item.samples.iter().enumerate() {
            let name = format!("{} #{}", canvas.name, n + 1);
            let prepared = nufo::training::prepare(sample, &canvas.contours)
                .unwrap_or_else(|e| panic!("{path}: {name}: {e}"));
            let clusters = prepared
                .letters
                .iter()
                .map(|ink| PhraseCluster {
                    letters: ink.letter.to_string(),
                    origin: None,
                    regions: ink
                        .regions
                        .iter()
                        .map(|poly| poly.iter().map(|p| [p.x, p.y]).collect())
                        .collect(),
                    paths: ink.contours.iter().map(|c| c.to_svg()).collect(),
                })
                .collect();
            out.push((
                name,
                Phrase {
                    text: prepared.text,
                    upm: source.units_per_em,
                    outline: Some(prepared.outline.to_svg()),
                    baseline_y: 0.0,
                    clusters,
                },
            ));
        }
    }
    println!("{path}: {} labeled sample(s)", out.len());
    out
}

pub fn hand(base_dir: &str, out_dir: &str, phrase_paths: &[String]) {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{base_dir}/fields-meta.json")).unwrap(),
    )
    .unwrap();
    let f = |k: &str| meta[k].as_f64().unwrap();
    let upm = f("upm");
    let mut cv = Canvas {
        w: f("w") as usize,
        h: f("h") as usize,
        origin_x: f("origin_x"),
        origin_y: f("origin_y"),
        px_per_unit: f("em_px") / upm,
        spread_px: f("spread_px"),
    };
    let base_shapes = f("shapes") as usize;
    std::fs::create_dir_all(out_dir).unwrap();

    let mut dataset = std::fs::read_to_string(format!("{base_dir}/dataset.jsonl")).unwrap();
    // An empty base has no rows: a font made from a source alone.
    if !dataset.is_empty() && !dataset.ends_with('\n') {
        dataset.push('\n');
    }
    let mut fields_bin = std::fs::read(format!("{base_dir}/fields.bin")).unwrap();
    let mut n_shapes = base_shapes;
    let mut space_ctx: Vec<[String; 4]> = Vec::new();
    let mut n_rows = 0usize;

    let mut phrases: Vec<(String, Phrase)> = Vec::new();
    for path in phrase_paths {
        if std::path::Path::new(path).is_dir() {
            phrases.extend(nufo_phrases(path));
        } else {
            let phrase: Phrase =
                serde_json::from_str(&std::fs::read_to_string(path).expect("phrase file")).unwrap();
            phrases.push((path.clone(), phrase));
        }
    }
    // A base with no shapes and `"auto_canvas": true` takes its canvas
    // from the phrases: large enough for the largest cluster's ink, with
    // the field's spread around it. The source decides the size, so a
    // longer letter needs no new setting.
    let mut meta = meta;
    if meta["auto_canvas"].as_bool() == Some(true) {
        assert_eq!(base_shapes, 0, "auto_canvas needs a base with no shapes");
        let (mut max_w, mut max_h) = (0.0f64, 0.0f64);
        let mut largest = String::new();
        for (path, phrase) in &phrases {
            let k = upm / phrase.upm;
            let outline =
                phrase.outline.as_deref().map(|d| kurbo::BezPath::from_svg(d).expect("outline"));
            for c in lay_out(path, phrase).1.iter().flatten() {
                if let Some(ink) = ink_box(c, outline.as_ref(), k, cv.px_per_unit) {
                    if ink.width() > max_w {
                        largest = format!("{} in {:?}", c.letters, phrase.text);
                    }
                    max_w = max_w.max(ink.width());
                    max_h = max_h.max(ink.height());
                }
            }
        }
        let margin = cv.spread_px + 2.0;
        cv.w = (max_w * cv.px_per_unit + 2.0 * margin).ceil() as usize;
        cv.h = (max_h * cv.px_per_unit + 2.0 * margin).ceil() as usize;
        cv.origin_x = cv.w as f64 / 2.0;
        cv.origin_y = cv.h as f64 / 2.0;
        meta["w"] = serde_json::json!(cv.w);
        meta["h"] = serde_json::json!(cv.h);
        meta["origin_x"] = serde_json::json!(cv.origin_x);
        meta["origin_y"] = serde_json::json!(cv.origin_y);
        println!(
            "canvas: {} x {} px ({:.0} x {:.0} units of ink at most; the widest is {largest})",
            cv.w, cv.h, max_w, max_h
        );
    }
    let cv = cv;

    // One empty field, shared by every space row.
    let empty_id = n_shapes;
    fields_bin.extend(sdf_from_grid(&vec![false; cv.w * SS * cv.h * SS], cv.w, cv.h, SS, cv.spread_px));
    n_shapes += 1;

    for (path, phrase) in &phrases {
        let path = path.as_str();
        let k = upm / phrase.upm;
        let outline = phrase.outline.as_deref().map(|d| kurbo::BezPath::from_svg(d).expect("outline"));
        let (words, laid) = lay_out(path, phrase);

        // Every cluster's origin in font units, given or centered.
        let origins: Vec<Vec<(f64, f64)>> = laid
            .iter()
            .map(|w| {
                w.iter()
                    .map(|c| {
                        c.origin
                            .map(|o| (o[0] * k, o[1] * k))
                            .or_else(|| centered_origin(c, outline.as_ref(), k, &cv))
                            .unwrap_or_else(|| panic!("{path}: cluster {:?} has no ink", c.letters))
                    })
                    .collect()
            })
            .collect();
        for (wi, (chars, clusters)) in words.iter().zip(&laid).enumerate() {
            // A space row chains this word to the previous one.
            if wi > 0 {
                let before = &words[wi - 1];
                let (px, py) = *origins[wi - 1].last().unwrap();
                let (nx, ny) = origins[wi][0];
                let n = before.len();
                let ctx = [
                    if n >= 2 { Some(before[n - 2]) } else { None },
                    before.last().copied(),
                    chars.first().copied(),
                    chars.get(1).copied(),
                ];
                dataset += &serde_json::json!({
                    "letters": " ", "prev2": ctx[0], "prev": ctx[1], "next": ctx[2], "next2": ctx[3],
                    "index": 0, "shape": empty_id, "pri": 1,
                    "ddx": (nx - px).round() as i32, "ddy": (ny - py).round() as i32,
                })
                .to_string();
                dataset.push('\n');
                n_rows += 1;
                let s = |c: Option<char>| c.map(String::from).unwrap_or_default();
                let key = [s(ctx[0]), s(ctx[1]), s(ctx[2]), s(ctx[3])];
                if !space_ctx.contains(&key) {
                    space_ctx.push(key);
                }
            }
            // The word origin only matters when the word is placed as
            // an ordinary word: left end of its origins, on the line.
            let wx = origins[wi].iter().map(|o| o.0).fold(f64::MAX, f64::min);
            let wy = phrase.baseline_y * k;
            let ranges = cluster_ranges(chars);
            let mut prev = (wx, wy);
            for (ci, (c, &(a, b))) in clusters.iter().zip(&ranges).enumerate() {
                let (ox, oy) = origins[wi][ci];
                let ink = cluster_ink(c, (ox, oy), outline.as_ref(), k, &cv);
                if !ink.iter().any(|&v| v) {
                    eprintln!("warning: {path}: cluster {:?} in {:?} has no ink", c.letters, phrase.text);
                }
                if clipped(&ink, cv.w * SS, cv.h * SS) {
                    eprintln!(
                        "warning: {path}: cluster {:?} in {:?} does not fit the canvas",
                        c.letters, phrase.text
                    );
                }
                fields_bin.extend(sdf_from_grid(&ink, cv.w, cv.h, SS, cv.spread_px));
                dataset += &serde_json::json!({
                    "letters": c.letters,
                    "prev2": if a >= 2 { Some(chars[a - 2]) } else { None },
                    "prev": if a >= 1 { Some(chars[a - 1]) } else { None },
                    "next": chars.get(b), "next2": chars.get(b + 1),
                    "index": ci, "shape": n_shapes, "pri": 1,
                    "ddx": (ox - prev.0).round() as i32, "ddy": (oy - prev.1).round() as i32,
                })
                .to_string();
                dataset.push('\n');
                prev = (ox, oy);
                n_shapes += 1;
                n_rows += 1;
            }
        }
    }

    std::fs::write(format!("{out_dir}/dataset.jsonl"), dataset).unwrap();
    std::fs::write(format!("{out_dir}/fields.bin"), &fields_bin).unwrap();
    meta["shapes"] = serde_json::json!(n_shapes);
    std::fs::write(
        format!("{out_dir}/fields-meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/header-extra.json"),
        serde_json::to_string_pretty(&serde_json::json!({ "space_ctx": space_ctx })).unwrap(),
    )
    .unwrap();
    println!(
        "wrote {out_dir}: {n_rows} labeled rows, {} new shapes, {} space contexts",
        n_shapes - base_shapes,
        space_ctx.len()
    );
}

/// Render a phrase file's own ink as a PGM, the ground truth to hold
/// a trained font against.
pub fn proof(phrase_path: &str, out_path: &str, em_px: f64) {
    let phrase: Phrase =
        serde_json::from_str(&std::fs::read_to_string(phrase_path).expect("phrase file")).unwrap();
    let mut paths: Vec<(kurbo::BezPath, f64, f64)> = Vec::new();
    if let Some(d) = &phrase.outline {
        paths.push((kurbo::BezPath::from_svg(d).expect("outline"), 0.0, 0.0));
    }
    for c in &phrase.clusters {
        for d in &c.paths {
            paths.push((kurbo::BezPath::from_svg(d).expect("cluster path"), 0.0, 0.0));
        }
    }
    let mut bb: Option<kurbo::Rect> = None;
    for (p, _, _) in &paths {
        if !p.elements().is_empty() {
            let r = kurbo::Shape::bounding_box(p);
            bb = Some(bb.map_or(r, |b| b.union(r)));
        }
    }
    let bb = bb.expect("phrase has no ink");
    let scale = em_px / phrase.upm;
    let pad = 4.0 / scale;
    let (w, h) = (
        ((bb.width() + 2.0 * pad) * scale).ceil() as usize,
        ((bb.height() + 2.0 * pad) * scale).ceil() as usize,
    );
    // One raster per path, OR-ed: clusters overlap at joins, and a
    // single nonzero fill would be right too, but this matches how
    // the engine composites.
    let mut grid = vec![false; w * h];
    for p in &paths {
        let g = rasterize(std::slice::from_ref(p), w, h, scale, bb.x0 - pad, bb.y1 + pad);
        for i in 0..grid.len() {
            grid[i] |= g[i];
        }
    }
    let mut f = std::fs::File::create(out_path).unwrap();
    writeln!(f, "P5\n{w} {h}\n255").unwrap();
    f.write_all(&grid.iter().map(|&b| if b { 255u8 } else { 0 }).collect::<Vec<u8>>()).unwrap();
    println!("wrote {out_path} ({w}x{h})");
}

/// Stand-in for a designer: shape a phrase with any OpenType font and
/// write it as a phrase file, each cluster with its own outline. Words
/// are placed as a simple stacked composition (each word tucked under
/// and raised above the one before), so the file exercises the space
/// rows. For testing the pipeline before real labeled data exists.
pub fn standin(font_path: &str, face: u32, text: &str, out_path: &str, size: f64) {
    let bytes = std::fs::read(font_path).expect("font not found");
    let font_ref = harfrust::FontRef::from_index(&bytes, face).expect("bad font");
    let shaper_data = harfrust::ShaperData::new(&font_ref);
    let shaper = shaper_data.shaper(&font_ref).build();
    let ttf = ttf_parser::Face::parse(&bytes, face).expect("bad font");
    let upm = ttf.units_per_em() as f64;

    struct B(kurbo::BezPath);
    impl ttf_parser::OutlineBuilder for B {
        fn move_to(&mut self, x: f32, y: f32) {
            self.0.move_to((x as f64, y as f64));
        }
        fn line_to(&mut self, x: f32, y: f32) {
            self.0.line_to((x as f64, y as f64));
        }
        fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
            self.0.quad_to((x1 as f64, y1 as f64), (x as f64, y as f64));
        }
        fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
            self.0.curve_to((x1 as f64, y1 as f64), (x2 as f64, y2 as f64), (x as f64, y as f64));
        }
        fn close(&mut self) {
            self.0.close_path();
        }
    }

    let mut clusters: Vec<PhraseCluster> = Vec::new();
    let (mut word_x, mut word_y) = (0.0f64, 0.0f64);
    for word in text.split(' ').filter(|w| !w.is_empty()) {
        let mut buf = harfrust::UnicodeBuffer::new();
        buf.push_str(word);
        buf.guess_segment_properties();
        let out = shaper.shape(buf, harfrust::ShapeOptions::default());
        let mut pen_x = 0i32;
        // cluster byte index -> placed glyphs, in buffer order
        let mut by_cluster: std::collections::BTreeMap<u32, Vec<(u32, f64, f64)>> = Default::default();
        for (info, pos) in out.glyph_infos().iter().zip(out.glyph_positions()) {
            by_cluster.entry(info.cluster).or_default().push((
                info.glyph_id,
                (pen_x + pos.x_offset) as f64,
                pos.y_offset as f64,
            ));
            pen_x += pos.x_advance;
        }
        // this word sits to the left of the previous one, tucked in
        // by a fifth of its width, and one step higher
        let advance = pen_x as f64 * size;
        word_x -= advance * 0.8;
        let keys: Vec<u32> = by_cluster.keys().copied().collect();
        for (ci, ck) in keys.iter().enumerate() {
            let end = keys.get(ci + 1).map(|&k| k as usize).unwrap_or(word.len());
            let letters = word[*ck as usize..end].to_string();
            let glyphs = &by_cluster[ck];
            let place = |x: f64, y: f64| (word_x + x * size, word_y + y * size);
            let origin = place(glyphs[0].1, glyphs[0].2);
            let paths = glyphs
                .iter()
                .filter_map(|&(gid, x, y)| {
                    let mut b = B(kurbo::BezPath::new());
                    ttf.outline_glyph(ttf_parser::GlyphId(gid as u16), &mut b)?;
                    let (tx, ty) = place(x, y);
                    let a = kurbo::Affine::translate((tx, ty)) * kurbo::Affine::scale(size);
                    Some((a * b.0).to_svg())
                })
                .collect();
            clusters.push(PhraseCluster {
                letters,
                origin: Some([origin.0, origin.1]),
                regions: Vec::new(),
                paths,
            });
        }
        word_y += 0.18 * upm;
    }
    let phrase = Phrase { text: text.to_string(), upm, outline: None, baseline_y: 0.0, clusters };
    std::fs::write(out_path, serde_json::to_string(&phrase).unwrap()).unwrap();
    println!(
        "wrote {out_path}: {} clusters: {}",
        phrase.clusters.len(),
        phrase.clusters.iter().map(|c| c.letters.as_str()).collect::<Vec<_>>().join(" ")
    );
}

/// List the faces of a font collection, to pick an index for standin.
pub fn faces(font_path: &str) {
    let bytes = std::fs::read(font_path).expect("font not found");
    let n = ttf_parser::fonts_in_collection(&bytes).unwrap_or(1);
    for i in 0..n {
        if let Ok(face) = ttf_parser::Face::parse(&bytes, i) {
            let name = face
                .names()
                .into_iter()
                .filter(|n| n.name_id == ttf_parser::name_id::FULL_NAME)
                .find_map(|n| n.to_string())
                .unwrap_or_default();
            println!("{i}  {name}  upm {}", face.units_per_em());
        }
    }
}
