//! Shaping and layout: Unicode text → joining forms → positioned,
//! model-generated glyph outlines.

use crate::art::{elongatable, joins_left, joins_right, letter_of_char, Class};
use crate::trace::trace_bitmap;
use crate::{Form, GlyphImage, GRID_H, GRID_W};
use kurbo::{BezPath, Shape};

/// Anything that can generate a glyph image for a character in
/// context: the procedural teacher, or the neural font.
pub trait GlyphSource {
    fn glyph(&self, c: char, form: Form, elong: f64) -> Option<GlyphImage>;
}

/// Cluster metadata: which character produced which horizontal span.
pub struct GlyphInfo {
    pub ch: char,
    pub form: Form,
    /// Left edge of the glyph's advance box in line units.
    pub x: f64,
    pub advance: f64,
}

/// Per-logical-character horizontal span, for cursor placement,
/// selection highlighting, and hit testing in editors.
pub struct Span {
    /// Logical (source-order) character index.
    pub index: usize,
    /// Left edge in line units.
    pub x: f64,
    pub width: f64,
}

pub struct Line {
    /// One outline for the whole line, traced from the composited
    /// bitmap of every generated glyph — connected letters are one
    /// continuous contour, exactly as the script demands. Font units
    /// (grid cells, y-down, baseline at `BASELINE_ROW`).
    pub path: BezPath,
    /// Total width of the line in font units.
    pub width: f64,
    /// Resolved base direction.
    pub rtl: bool,
    pub glyphs: Vec<GlyphInfo>,
    /// One span per source character (letters, spaces, unknowns).
    pub spans: Vec<Span>,
}

/// Width of a word space, in cells.
const SPACE_ADV: f64 = 3.0;
/// Gap between unconnected letters, in cells.
const LETTER_GAP: f64 = 1.0;

/// Base paragraph direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    /// RTL if the text contains any Arabic letter, else LTR.
    Auto,
    Rtl,
    Ltr,
}

/// Resolve the visual character order of a line (bidi-lite), keeping
/// each character's logical index for cluster mapping. The layout pen
/// always runs right-to-left, so: RTL text keeps its logical order
/// with Latin runs reversed in place; LTR text is reversed wholesale,
/// which yields left-to-right rendering. Returns (visual chars, rtl).
fn visual_order(text: &str, dir: Dir) -> (Vec<(usize, char)>, bool) {
    let mut chars: Vec<(usize, char)> = text.chars().enumerate().collect();
    let is_latin = |c: char| matches!(letter_of_char(c), Some((Class::Latin(_), _)));
    let is_arabic = |c: char| matches!(letter_of_char(c), Some((cl, _)) if !matches!(cl, Class::Latin(_)));
    let rtl = match dir {
        Dir::Rtl => true,
        Dir::Ltr => false,
        Dir::Auto => chars.iter().any(|&(_, c)| is_arabic(c)),
    };
    if !rtl {
        chars.reverse();
        return (chars, rtl);
    }
    // Reverse each maximal run of Latin letters (spaces break runs).
    let mut i = 0;
    while i < chars.len() {
        if is_latin(chars[i].1) {
            let mut j = i;
            while j < chars.len() && is_latin(chars[j].1) {
                j += 1;
            }
            chars[i..j].reverse();
            i = j;
        } else {
            i += 1;
        }
    }
    (chars, rtl)
}

/// Compute the joining form of each letter in `text`, in visual order,
/// keeping logical indices (a glyph may carry two — the لا ligature).
/// Non-letters (spaces, unsupported chars) and non-joining scripts
/// break joining.
pub fn shaped_forms(text: &str, dir: Dir) -> (Vec<(Vec<usize>, char, Option<Form>)>, bool) {
    let (chars, rtl) = visual_order(text, dir);
    // Cluster pass: mandatory لا ligature — lam followed by alef fuses
    // into one glyph carrying both logical indices.
    let mut clusters: Vec<(Vec<usize>, char)> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let (li, c) = chars[i];
        let lam = matches!(letter_of_char(c), Some((Class::Lam, _)));
        let next_alef = i + 1 < chars.len()
            && matches!(letter_of_char(chars[i + 1].1), Some((Class::Alef, _)));
        if lam && next_alef {
            clusters.push((vec![li, chars[i + 1].0], '\u{FEFB}')); // ﻻ
            i += 2;
        } else {
            clusters.push((vec![li], c));
            i += 1;
        }
    }
    let is_letter = |i: i32| -> bool {
        i >= 0 && (i as usize) < clusters.len() && letter_of_char(clusters[i as usize].1).is_some()
    };
    let joins_to_next = |i: usize| -> bool {
        // glyph i connects to glyph i+1
        is_letter(i as i32)
            && is_letter(i as i32 + 1)
            && joins_left(letter_of_char(clusters[i].1).unwrap().0)
            && joins_right(letter_of_char(clusters[i + 1].1).unwrap().0)
    };
    let shaped = clusters
        .iter()
        .enumerate()
        .map(|(i, (lis, c))| {
            if letter_of_char(*c).is_none() {
                return (lis.clone(), *c, None);
            }
            let prev = i > 0 && joins_to_next(i - 1);
            let next = joins_to_next(i);
            let form = match (prev, next) {
                (false, false) => Form::Isolated,
                (false, true) => Form::Initial,
                (true, true) => Form::Medial,
                (true, false) => Form::Final,
            };
            (lis.clone(), *c, Some(form))
        })
        .collect();
    (shaped, rtl)
}

/// Shape and lay out a line of text right-to-left.
/// `elong` in [0, MAX_ELONG] is the kashida elongation applied at
/// every connection — a continuous input to the generative font.
pub fn layout(source: &dyn GlyphSource, text: &str, elong: f64, dir: Dir) -> Line {
    // Kashida in a grid style is a whole number of cells; the model is
    // trained at integer levels, so snap. (Truly continuous elongation
    // arrives with the bezier output head — see docs/SPEC.md.)
    let elong = elong.round();
    let (shaped, rtl) = shaped_forms(text, dir);
    // Pass 1: generate every glyph and record pen positions (the pen
    // moves leftward from 0; advances are whole cells). Every source
    // character — letter, space, or unknown — gets a span for cursor
    // placement and selection.
    let mut placed: Vec<(GlyphImage, f64, char, Form)> = Vec::new();
    let mut spans: Vec<Span> = Vec::new();
    let mut pen_x = 0.0;
    for (i, (lis, c, form)) in shaped.iter().enumerate() {
        let (c, form) = (*c, *form);
        let Some(form) = form else {
            // Spaces (and unsupported chars) advance the pen.
            for &li in lis {
                spans.push(Span { index: li, x: pen_x - SPACE_ADV, width: SPACE_ADV });
            }
            pen_x -= SPACE_ADV;
            continue;
        };
        // Elongate where the glyph supports it: Arabic connections
        // (kashida) or Latin stretch strokes.
        let class = letter_of_char(c).unwrap().0;
        let e = if elongatable(class, form) { elong } else { 0.0 };
        let Some(img) = source.glyph(c, form, e) else {
            for &li in lis {
                spans.push(Span { index: li, x: pen_x, width: 0.0 });
            }
            continue;
        };
        let adv = img.advance;
        placed.push((img, pen_x, c, form));
        // A ligature's advance is split across its source characters
        // (visual order = right to left within the glyph).
        let n = lis.len() as f64;
        for (k, &li) in lis.iter().enumerate() {
            let piece = adv / n;
            spans.push(Span {
                index: li,
                x: pen_x - (k as f64 + 1.0) * piece,
                width: piece,
            });
        }
        pen_x -= adv;
        // Gap after a letter that does not connect to the next letter.
        let connects = shaped
            .get(i + 1)
            .map(|(_, _, f)| matches!(f, Some(Form::Medial) | Some(Form::Final)))
            .unwrap_or(false);
        if !connects && i + 1 < shaped.len() {
            pen_x -= LETTER_GAP;
        }
    }
    let width = -pen_x;
    for s in &mut spans {
        s.x += width;
    }
    spans.sort_by_key(|s| s.index);

    // Pass 2: composite all glyph bitmaps into one line-wide grid and
    // trace it once, so connections merge into continuous contours.
    let w_cells = width.round().max(1.0) as usize;
    let mut cells = vec![false; w_cells * GRID_H];
    let mut glyphs = Vec::new();
    for (img, pen_right, c, form) in &placed {
        // The glyph canvas's right edge sits at pen_right.
        let x_off = (width + pen_right).round() as i64 - GRID_W as i64;
        for cy in 0..GRID_H {
            for cx in 0..GRID_W {
                if img.get(cx, cy) {
                    let lx = x_off + cx as i64;
                    if lx >= 0 && (lx as usize) < w_cells {
                        cells[cy * w_cells + lx as usize] = true;
                    }
                }
            }
        }
        glyphs.push(GlyphInfo {
            ch: *c,
            form: *form,
            x: width + pen_right - img.advance,
            advance: img.advance,
        });
    }
    let path = trace_bitmap(w_cells, GRID_H, |x, y| cells[y * w_cells + x]);
    Line { path, width, rtl, glyphs, spans }
}

/// Convenience: the line's outline as an SVG path string.
pub fn line_to_svg_path(line: &Line) -> String {
    line.path.to_path(1e-3).to_svg()
}
