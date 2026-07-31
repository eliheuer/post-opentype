//! distill: extract a (context → shape) dataset from an OpenType font.
//!
//! This is step 1 of the TTF→NTF conversion (docs/DISTILL.md). The
//! font's own shaping engine output is the teacher: we shape a corpus
//! with harfrust (the HarfBuzz-org Rust port, HarfBuzz 13 parity), group the shaped glyphs into per-letter clusters
//! (a base glyph plus its attached marks), and record every cluster
//! with its context and its displacement from the previous cluster.
//! In nastaliq the displacement chain IS the cascade.
//!
//! Usage:
//!   distill extract <font.ttf> <out-dir>   shape the corpus, write dataset
//!   distill stats <out-dir>                summarize an extracted dataset
//!   distill fields <extract-dir> <out-dir> [em_px]   render SDF fields
//!   distill proof <fields-dir> <out.pgm> [ids...]    field proof sheet
//!
//! Output files (JSONL, one record per line, inspectable with jq):
//!   meta.json      font name, upm, corpus and dedup counts
//!   glyphs.jsonl   {gid, path} for every glyph the corpus touched
//!   contexts.jsonl one record per cluster occurrence, see ClusterRec

mod fields;

use neuraltype_core::art::letter_of_char;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::io::Write as _;

/// The Arabic letters the corpus is built from.
const LETTERS: &[char] = &[
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ',
    'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'ة', 'و', 'ي', 'ى', 'ء',
];

/// One glyph placed inside a cluster, relative to the cluster origin.
#[derive(Serialize, Deserialize)]
struct PlacedGlyph {
    gid: u32,
    dx: i32,
    dy: i32,
}

/// One cluster occurrence: a letter (or ligature) as shaped in one word.
#[derive(Serialize, Deserialize)]
struct ClusterRec {
    /// The whole source word.
    word: String,
    /// The characters this cluster covers (usually one, لا is two).
    letters: String,
    /// Cluster index in logical order within the word.
    index: usize,
    /// Letters immediately before/after in logical order, if any.
    prev: Option<char>,
    next: Option<char>,
    /// Second-order neighbors; Gulzar's rules reach this far.
    prev2: Option<char>,
    next2: Option<char>,
    /// Base glyph and marks, placed relative to the cluster origin.
    glyphs: Vec<PlacedGlyph>,
    /// Displacement from the previous cluster's origin to this one
    /// (font units). None for the first cluster of the word.
    ddx: Option<i32>,
    ddy: Option<i32>,
    /// This cluster's origin relative to the word origin.
    ox: i32,
    oy: i32,
}

#[derive(Serialize)]
struct Meta {
    font: String,
    upm: u16,
    words: usize,
    clusters: usize,
    unique_glyphs: usize,
    unique_cluster_shapes: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("extract") => extract(
            args.get(1).expect("usage: distill extract <font.ttf> <out-dir>"),
            args.get(2).expect("usage: distill extract <font.ttf> <out-dir>"),
        ),
        Some("stats") => stats(args.get(1).expect("usage: distill stats <out-dir>")),
        Some("fields") => fields::fields(
            args.get(1).expect("usage: distill fields <extract-dir> <out-dir> [em_px]"),
            args.get(2).expect("usage: distill fields <extract-dir> <out-dir> [em_px]"),
            args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64),
        ),
        Some("proof") => fields::proof(
            args.get(1).expect("usage: distill proof <fields-dir> <out.pgm> [ids...]"),
            args.get(2).expect("usage: distill proof <fields-dir> <out.pgm> [ids...]"),
            &args[3..].iter().filter_map(|s| s.parse().ok()).collect::<Vec<usize>>(),
        ),
        _ => eprintln!("usage: distill extract|stats|fields|proof ..."),
    }
}

/// Corpus: every single letter, every ordered pair, every ordered
/// triple. Triples are what capture medial forms in their full
/// (previous, next) context.
fn corpus() -> Vec<String> {
    let mut words = Vec::new();
    for &a in LETTERS {
        words.push(a.to_string());
    }
    for &a in LETTERS {
        for &b in LETTERS {
            words.push([a, b].iter().collect());
        }
    }
    for &a in LETTERS {
        for &b in LETTERS {
            for &c in LETTERS {
                words.push([a, b, c].iter().collect());
            }
        }
    }
    words
}

/// Extract an SVG path for a glyph, in font units (y-up).
struct PathBuilder(String);
impl ttf_parser::OutlineBuilder for PathBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        let _ = write!(self.0, "M{x} {y}");
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let _ = write!(self.0, "L{x} {y}");
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let _ = write!(self.0, "Q{x1} {y1} {x} {y}");
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let _ = write!(self.0, "C{x1} {y1} {x2} {y2} {x} {y}");
    }
    fn close(&mut self) {
        self.0.push('Z');
    }
}

fn extract(font_path: &str, out_dir: &str) {
    let bytes = std::fs::read(font_path).expect("font not found");
    let font_ref = harfrust::FontRef::from_index(&bytes, 0).expect("bad font");
    let shaper_data = harfrust::ShaperData::new(&font_ref);
    let shaper = shaper_data.shaper(&font_ref).build();
    let ttf = ttf_parser::Face::parse(&bytes, 0).expect("bad font");
    std::fs::create_dir_all(out_dir).unwrap();

    let words = corpus();
    println!("corpus: {} words", words.len());

    let mut used_glyphs: HashSet<u32> = HashSet::new();
    let mut shape_dedup: HashSet<String> = HashSet::new();
    let mut n_clusters = 0usize;
    let ctx_file = std::fs::File::create(format!("{out_dir}/contexts.jsonl")).unwrap();
    let mut ctx = std::io::BufWriter::new(ctx_file);

    for (wi, word) in words.iter().enumerate() {
        if wi % 5000 == 0 {
            println!("  {wi}/{}", words.len());
        }
        let mut buf = harfrust::UnicodeBuffer::new();
        buf.push_str(word);
        buf.guess_segment_properties();
        let out = shaper.shape(buf, harfrust::ShapeOptions::default());
        let infos = out.glyph_infos();
        let positions = out.glyph_positions();

        // Absolute drawing position of every glyph (pen accumulation
        // in buffer order, which is visual order for RTL).
        let mut pen_x = 0i32;
        let mut placed: Vec<(u32, u32, i32, i32)> = Vec::new(); // gid, cluster, x, y
        for (info, pos) in infos.iter().zip(positions) {
            placed.push((
                info.glyph_id,
                info.cluster,
                pen_x + pos.x_offset,
                pos.y_offset,
            ));
            pen_x += pos.x_advance;
        }

        // Group by cluster (byte index), then order logically.
        let mut by_cluster: BTreeMap<u32, Vec<(u32, i32, i32)>> = BTreeMap::new();
        for (gid, cl, x, y) in &placed {
            by_cluster.entry(*cl).or_default().push((*gid, *x, *y));
            used_glyphs.insert(*gid);
        }
        let chars: Vec<(usize, char)> = word.char_indices().collect();
        let cluster_keys: Vec<u32> = by_cluster.keys().cloned().collect();

        let mut prev_origin: Option<(i32, i32)> = None;
        for (ci, &ck) in cluster_keys.iter().enumerate() {
            let glyphs = &by_cluster[&ck];
            // Origin = position of the cluster's base glyph (the first
            // glyph in visual placement; marks follow their base).
            let (bx, by) = (glyphs[0].1, glyphs[0].2);
            let rel: Vec<PlacedGlyph> = glyphs
                .iter()
                .map(|(g, x, y)| PlacedGlyph { gid: *g, dx: x - bx, dy: y - by })
                .collect();
            // Letters covered: chars in [ck, next cluster key).
            let end = cluster_keys.get(ci + 1).map(|&k| k as usize).unwrap_or(word.len());
            let letters: String = chars
                .iter()
                .filter(|(i, _)| *i >= ck as usize && *i < end)
                .map(|(_, c)| c)
                .collect();
            let before: Vec<char> =
                chars.iter().rev().filter(|(i, _)| *i < ck as usize).map(|(_, c)| *c).collect();
            let after: Vec<char> =
                chars.iter().filter(|(i, _)| *i >= end).map(|(_, c)| *c).collect();
            let prev = before.first().copied();
            let prev2 = before.get(1).copied();
            let next = after.first().copied();
            let next2 = after.get(1).copied();

            let (ddx, ddy) = match prev_origin {
                Some((px, py)) => (Some(bx - px), Some(by - py)),
                None => (None, None),
            };
            prev_origin = Some((bx, by));

            // Dedup key for counting distinct cluster shapes.
            let key = serde_json::to_string(&rel).unwrap();
            shape_dedup.insert(format!("{letters}|{key}"));

            let rec = ClusterRec {
                word: word.clone(),
                letters,
                index: ci,
                prev,
                next,
                prev2,
                next2,
                glyphs: rel,
                ddx,
                ddy,
                ox: bx,
                oy: by,
            };
            writeln!(ctx, "{}", serde_json::to_string(&rec).unwrap()).unwrap();
            n_clusters += 1;
        }
    }

    // Outlines for every glyph the corpus touched.
    let glyph_file = std::fs::File::create(format!("{out_dir}/glyphs.jsonl")).unwrap();
    let mut gw = std::io::BufWriter::new(glyph_file);
    let mut gids: Vec<u32> = used_glyphs.iter().cloned().collect();
    gids.sort();
    for gid in &gids {
        let mut b = PathBuilder(String::new());
        ttf.outline_glyph(ttf_parser::GlyphId(*gid as u16), &mut b);
        writeln!(gw, "{}", serde_json::json!({ "gid": gid, "path": b.0 })).unwrap();
    }

    let meta = Meta {
        font: font_path.into(),
        upm: ttf.units_per_em(),
        words: words.len(),
        clusters: n_clusters,
        unique_glyphs: gids.len(),
        unique_cluster_shapes: shape_dedup.len(),
    };
    std::fs::write(
        format!("{out_dir}/meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
    println!("{}", serde_json::to_string_pretty(&meta).unwrap());
}

fn stats(out_dir: &str) {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{out_dir}/meta.json")).unwrap())
            .unwrap();
    println!("{}", serde_json::to_string_pretty(&meta).unwrap());

    // Displacement distribution and per-letter shape variety.
    let mut dy_min = i32::MAX;
    let mut dy_max = i32::MIN;
    let mut shapes_per_letter: HashMap<String, HashSet<String>> = HashMap::new();
    let file = std::fs::read_to_string(format!("{out_dir}/contexts.jsonl")).unwrap();
    for line in file.lines() {
        let r: ClusterRec = serde_json::from_str(line).unwrap();
        if let Some(dy) = r.ddy {
            dy_min = dy_min.min(dy);
            dy_max = dy_max.max(dy);
        }
        let key = serde_json::to_string(&r.glyphs).unwrap();
        shapes_per_letter.entry(r.letters.clone()).or_default().insert(key);
    }
    println!();
    println!("cluster displacement dy range: {dy_min} .. {dy_max} font units");
    println!();
    println!("distinct cluster shapes per letter (composed, marks included):");
    let mut rows: Vec<(String, usize)> = shapes_per_letter
        .iter()
        .map(|(k, v)| (k.clone(), v.len()))
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    for (letters, n) in rows.iter().take(15) {
        let known = letters.chars().all(|c| letter_of_char(c).is_some());
        println!("  {letters}  {n}{}", if known { "" } else { "  (unmapped)" });
    }
    let total: usize = rows.iter().map(|(_, n)| n).sum();
    println!("  total distinct cluster shapes: {total}");
}
