//! Trace a glyph occupancy grid into rectilinear bezier outlines.
//!
//! Each filled cell contributes its exposed edges, oriented so the
//! filled region is on the left; edges are chained into closed loops
//! and emitted as a kurbo `BezPath` (outer contours clockwise in
//! screen coordinates, holes counter-clockwise — nonzero fill works).
//! Coordinates are in grid-cell units, y-down, cell (x, y) covering
//! [x, x+1] × [y, y+1].

use crate::{GlyphImage, GRID_H, GRID_W};
use kurbo::BezPath;
use std::collections::HashMap;

type Pt = (i32, i32);

pub fn trace_outline(img: &GlyphImage) -> BezPath {
    trace_bitmap(GRID_W, GRID_H, |x, y| img.get(x, y))
}

/// Trace any bitmap — in particular a whole composited line, so that
/// connected letters become one continuous contour, not per-glyph
/// rectangles that merely abut.
pub fn trace_bitmap(w: usize, h: usize, cell: impl Fn(usize, usize) -> bool) -> BezPath {
    // Collect directed boundary edges: filled area on the left.
    let mut by_start: HashMap<Pt, Vec<Pt>> = HashMap::new();
    let mut add = |a: Pt, b: Pt| by_start.entry(a).or_default().push(b);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            if !cell(x as usize, y as usize) {
                continue;
            }
            let filled = |x: i32, y: i32| {
                x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h
                    && cell(x as usize, y as usize)
            };
            if !filled(x, y - 1) {
                add((x, y), (x + 1, y)); // top edge, heading right
            }
            if !filled(x, y + 1) {
                add((x + 1, y + 1), (x, y + 1)); // bottom edge, heading left
            }
            if !filled(x - 1, y) {
                add((x, y + 1), (x, y)); // left edge, heading up
            }
            if !filled(x + 1, y) {
                add((x + 1, y), (x + 1, y + 1)); // right edge, heading down
            }
        }
    }

    // Chain edges into loops. At checkerboard corners two edges share
    // a start point; prefer the sharpest right turn so touching
    // regions stay separate loops.
    let mut path = BezPath::new();
    loop {
        let Some(&start) = by_start.keys().next() else { break };
        let mut pts: Vec<Pt> = vec![start];
        let mut cur = start;
        let mut dir: Pt = (0, 0);
        loop {
            let outs = by_start.get_mut(&cur).unwrap();
            let next = if outs.len() == 1 {
                outs[0]
            } else {
                // pick the candidate making the sharpest right turn
                // relative to the incoming direction (y-down coords).
                *outs
                    .iter()
                    .max_by_key(|&&(nx, ny)| {
                        let nd = (nx - cur.0, ny - cur.1);
                        // cross product dir × nd; right turn (screen) > 0
                        dir.0 * nd.1 - dir.1 * nd.0
                    })
                    .unwrap()
            };
            outs.retain(|&p| p != next);
            if outs.is_empty() {
                by_start.remove(&cur);
            }
            dir = (next.0 - cur.0, next.1 - cur.1);
            cur = next;
            if cur == start {
                break;
            }
            pts.push(cur);
        }
        emit_loop(&mut path, &pts);
    }
    path
}

fn emit_loop(path: &mut BezPath, pts: &[Pt]) {
    if pts.len() < 4 {
        return;
    }
    // Merge collinear runs.
    let n = pts.len();
    let mut corners: Vec<Pt> = Vec::new();
    for i in 0..n {
        let prev = pts[(i + n - 1) % n];
        let next = pts[(i + 1) % n];
        let a = (pts[i].0 - prev.0, pts[i].1 - prev.1);
        let b = (next.0 - pts[i].0, next.1 - pts[i].1);
        if a.0 * b.1 - a.1 * b.0 != 0 {
            corners.push(pts[i]);
        }
    }
    if corners.len() < 4 {
        return;
    }
    path.move_to((corners[0].0 as f64, corners[0].1 as f64));
    for &(x, y) in &corners[1..] {
        path.line_to((x as f64, y as f64));
    }
    path.close_path();
}
