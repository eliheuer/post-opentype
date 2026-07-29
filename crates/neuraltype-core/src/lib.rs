//! neuraltype-core: a post-OpenType generative font engine.
//!
//! A font here is not a table of outlines — it is a small neural model
//! that generates glyph shapes on the fly, conditioned on character
//! identity, joining context, and continuous parameters (elongation).
//!
//! v0 targets square Kufic: glyphs are occupancy patterns on a coarse
//! grid, traced to rectilinear bezier outlines with kurbo. The same
//! pipeline is designed to graduate to naskh/nastaliq by swapping the
//! output representation from an occupancy grid to bezier control
//! points (see docs/SPEC.md).

pub mod art;
pub mod font;
pub mod model;
pub mod shape;
pub mod trace;

/// Glyph canvas dimensions, in grid cells.
pub const GRID_W: usize = 16;
pub const GRID_H: usize = 14;
/// Row index of the baseline stroke (letters sit on this row;
/// rows below are the descender zone).
pub const BASELINE_ROW: usize = 10;
/// Maximum kashida elongation, in extra baseline columns.
pub const MAX_ELONG: usize = 4;

/// The four contextual joining forms of the Arabic script.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Form {
    Isolated,
    Initial,
    Medial,
    Final,
}

impl Form {
    pub const ALL: [Form; 4] = [Form::Isolated, Form::Initial, Form::Medial, Form::Final];
    pub fn index(self) -> usize {
        match self {
            Form::Isolated => 0,
            Form::Initial => 1,
            Form::Medial => 2,
            Form::Final => 3,
        }
    }
    /// Does this form connect to the preceding letter (on its right)?
    pub fn joins_prev(self) -> bool {
        matches!(self, Form::Medial | Form::Final)
    }
}

/// A glyph image: binary occupancy grid plus advance width, all in
/// grid-cell units. This is both the teacher's output and the model's
/// (thresholded) output.
#[derive(Clone, Debug)]
pub struct GlyphImage {
    /// Row-major, GRID_H rows of GRID_W cells. Column 0 is the LEFT
    /// edge of the canvas; the glyph body is right-aligned (Arabic
    /// enters from the right).
    pub cells: Vec<bool>,
    /// Advance width in cells (how far the pen moves left).
    pub advance: f64,
}

impl GlyphImage {
    pub fn empty() -> Self {
        GlyphImage { cells: vec![false; GRID_W * GRID_H], advance: 0.0 }
    }
    pub fn get(&self, x: usize, y: usize) -> bool {
        x < GRID_W && y < GRID_H && self.cells[y * GRID_W + x]
    }
    pub fn set(&mut self, x: usize, y: usize) {
        if x < GRID_W && y < GRID_H {
            self.cells[y * GRID_W + x] = true;
        }
    }
}
