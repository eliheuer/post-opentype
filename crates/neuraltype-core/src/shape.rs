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

pub struct Line {
    /// One outline for the whole line, traced from the composited
    /// bitmap of every generated glyph — connected letters are one
    /// continuous contour, exactly as the script demands. Font units
    /// (grid cells, y-down, baseline at `BASELINE_ROW`).
    pub path: BezPath,
    /// Total width of the line in font units.
    pub width: f64,
    pub glyphs: Vec<GlyphInfo>,
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

/// Resolve the visual character order of a line (bidi-lite).
/// The layout pen always runs right-to-left, so: RTL text keeps its
/// logical order with Latin runs reversed in place; LTR text is
/// reversed wholesale, which yields left-to-right rendering.
fn visual_order(text: &str, dir: Dir) -> Vec<char> {
    let mut chars: Vec<char> = text.chars().collect();
    let is_latin = |c: char| matches!(letter_of_char(c), Some((Class::Latin(_), _)));
    let is_arabic = |c: char| matches!(letter_of_char(c), Some((cl, _)) if !matches!(cl, Class::Latin(_)));
    let rtl = match dir {
        Dir::Rtl => true,
        Dir::Ltr => false,
        Dir::Auto => chars.iter().any(|&c| is_arabic(c)),
    };
    if !rtl {
        chars.reverse();
        return chars;
    }
    // Reverse each maximal run of Latin letters (spaces break runs).
    let mut i = 0;
    while i < chars.len() {
        if is_latin(chars[i]) {
            let mut j = i;
            while j < chars.len() && is_latin(chars[j]) {
                j += 1;
            }
            chars[i..j].reverse();
            i = j;
        } else {
            i += 1;
        }
    }
    chars
}

/// Compute the joining form of each letter in `text` (already in
/// visual order). Non-letters (spaces, unsupported chars) and
/// non-joining scripts break joining.
pub fn shaped_forms(text: &str, dir: Dir) -> Vec<(char, Option<Form>)> {
    let chars = visual_order(text, dir);
    let is_letter = |i: i32| -> bool {
        i >= 0 && (i as usize) < chars.len() && letter_of_char(chars[i as usize]).is_some()
    };
    let joins_to_next = |i: usize| -> bool {
        // letter i connects to letter i+1
        is_letter(i as i32)
            && is_letter(i as i32 + 1)
            && joins_left(letter_of_char(chars[i]).unwrap().0)
            && joins_right(letter_of_char(chars[i + 1]).unwrap().0)
    };
    chars
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            if letter_of_char(c).is_none() {
                return (c, None);
            }
            let prev = i > 0 && joins_to_next(i - 1);
            let next = joins_to_next(i);
            let form = match (prev, next) {
                (false, false) => Form::Isolated,
                (false, true) => Form::Initial,
                (true, true) => Form::Medial,
                (true, false) => Form::Final,
            };
            (c, Some(form))
        })
        .collect()
}

/// Shape and lay out a line of text right-to-left.
/// `elong` in [0, MAX_ELONG] is the kashida elongation applied at
/// every connection — a continuous input to the generative font.
pub fn layout(source: &dyn GlyphSource, text: &str, elong: f64, dir: Dir) -> Line {
    let shaped = shaped_forms(text, dir);
    // Pass 1: generate every glyph and record pen positions (the pen
    // moves leftward from 0; advances are whole cells).
    let mut placed: Vec<(GlyphImage, f64, char, Form)> = Vec::new();
    let mut pen_x = 0.0;
    for (i, &(c, form)) in shaped.iter().enumerate() {
        let Some(form) = form else {
            pen_x -= SPACE_ADV;
            continue;
        };
        // Elongate where the glyph supports it: Arabic connections
        // (kashida) or Latin stretch strokes.
        let class = letter_of_char(c).unwrap().0;
        let e = if elongatable(class, form) { elong } else { 0.0 };
        let Some(img) = source.glyph(c, form, e) else { continue };
        let adv = img.advance;
        placed.push((img, pen_x, c, form));
        pen_x -= adv;
        // Gap after a letter that does not connect to the next letter.
        let connects = shaped
            .get(i + 1)
            .map(|&(_, f)| matches!(f, Some(Form::Medial) | Some(Form::Final)))
            .unwrap_or(false);
        if !connects && i + 1 < shaped.len() {
            pen_x -= LETTER_GAP;
        }
    }
    let width = -pen_x;

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
    Line { path, width, glyphs }
}

/// Convenience: the line's outline as an SVG path string.
pub fn line_to_svg_path(line: &Line) -> String {
    line.path.to_path(1e-3).to_svg()
}
