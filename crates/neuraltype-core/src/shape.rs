//! Shaping and layout: Unicode text → joining forms → positioned,
//! model-generated glyph outlines.

use crate::art::{joins_left, letter_of_char};
use crate::trace::trace_outline;
use crate::{Form, GlyphImage, GRID_W};
use kurbo::{Affine, BezPath, Shape};

/// Anything that can generate a glyph image for a character in
/// context: the procedural teacher, or the neural font.
pub trait GlyphSource {
    fn glyph(&self, c: char, form: Form, elong: f64) -> Option<GlyphImage>;
}

pub struct PositionedGlyph {
    pub ch: char,
    pub form: Form,
    /// Outline in font units (grid cells, y-down, baseline at
    /// `BASELINE_ROW`), already translated to its position in the line.
    pub path: BezPath,
}

pub struct Line {
    pub glyphs: Vec<PositionedGlyph>,
    /// Total width of the line in font units.
    pub width: f64,
}

/// Width of a word space, in cells.
const SPACE_ADV: f64 = 3.0;
/// Gap between unconnected letters, in cells.
const LETTER_GAP: f64 = 1.0;

/// Compute the joining form of each Arabic letter in `text`.
/// Non-letters (spaces, unsupported chars) break joining.
pub fn shaped_forms(text: &str) -> Vec<(char, Option<Form>)> {
    let chars: Vec<char> = text.chars().collect();
    let is_letter = |i: i32| -> bool {
        i >= 0 && (i as usize) < chars.len() && letter_of_char(chars[i as usize]).is_some()
    };
    let joins_to_next = |i: usize| -> bool {
        // letter i connects to letter i+1
        is_letter(i as i32)
            && is_letter(i as i32 + 1)
            && joins_left(letter_of_char(chars[i]).unwrap().0)
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
pub fn layout(source: &dyn GlyphSource, text: &str, elong: f64) -> Line {
    let shaped = shaped_forms(text);
    let mut glyphs = Vec::new();
    let mut pen_x = 0.0; // moves leftward (negative)
    for (i, &(c, form)) in shaped.iter().enumerate() {
        let Some(form) = form else {
            pen_x -= SPACE_ADV;
            continue;
        };
        // Only elongate connections (forms that join the previous letter).
        let e = if form.joins_prev() { elong } else { 0.0 };
        let Some(img) = source.glyph(c, form, e) else { continue };
        // Canvas right edge sits at pen_x; translate traced outline.
        let outline = trace_outline(&img);
        let path = Affine::translate((pen_x - GRID_W as f64, 0.0)) * outline;
        glyphs.push(PositionedGlyph { ch: c, form, path });
        pen_x -= img.advance;
        // Gap after a letter that does not connect to the next letter.
        let connects = shaped
            .get(i + 1)
            .map(|&(_, f)| matches!(f, Some(Form::Medial) | Some(Form::Final)))
            .unwrap_or(false);
        if !connects && i + 1 < shaped.len() {
            pen_x -= LETTER_GAP;
        }
    }
    // Shift so the line starts at x = 0.
    let width = -pen_x;
    let mut line = Line { glyphs, width };
    for g in &mut line.glyphs {
        g.path = Affine::translate((width, 0.0)) * g.path.clone();
    }
    line
}

/// Convenience: one combined SVG path string for a line.
pub fn line_to_svg_path(line: &Line) -> String {
    let mut s = String::new();
    for g in &line.glyphs {
        s.push_str(&g.path.to_path(1e-3).to_svg());
        s.push(' ');
    }
    s
}
