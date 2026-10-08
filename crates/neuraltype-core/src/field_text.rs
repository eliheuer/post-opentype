//! Text layout for field fonts: cluster segmentation, context
//! features, origin chaining (the cascade), word compositing, and
//! sub-pixel contour extraction.

use crate::field_model::FieldFont;
use kurbo::BezPath;
use std::collections::HashMap;

/// One laid-out cluster: its feature tuple and its origin in pixels
/// (y-down, baseline at y = 0).
#[derive(Clone)]
pub struct Cluster {
    pub letters: String,
    pub feats: [u32; 5],
    pub ox: f64,
    pub oy: f64,
}

/// Segment a word's chars into clusters (لا fuses) and build the
/// (prev2, prev, letter, next, next2) feature tuple for each.
pub fn word_clusters(font: &FieldFont, chars: &[char]) -> Vec<(String, [u32; 5])> {
    let ranges = cluster_ranges(chars);
    let none = font.none_id();
    let id_char = |c: Option<char>| -> u32 {
        c.and_then(|c| font.vocab_id(&c.to_string())).unwrap_or(none)
    };
    ranges
        .iter()
        .map(|&(a, b)| {
            let letters: String = chars[a..b].iter().collect();
            let feats = [
                id_char(if a >= 2 { Some(chars[a - 2]) } else { None }),
                id_char(if a >= 1 { Some(chars[a - 1]) } else { None }),
                font.vocab_id(&letters).unwrap_or(none),
                id_char(chars.get(b).copied()),
                id_char(chars.get(b + 1).copied()),
            ];
            (letters, feats)
        })
        .collect()
}

/// The char ranges of a word's clusters. Font-independent, so dataset
/// builders segment exactly as the engine does.
pub fn cluster_ranges(chars: &[char]) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    if chars == ['\u{627}', '\u{644}', '\u{644}', '\u{647}'] {
        // Gulzar ligates لله after the alif in the word الله: harfbuzz
        // shapes it as [ا][لله]. Match the teacher's clustering.
        ranges.push((0, 1));
        ranges.push((1, 4));
    } else {
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == 'ل' && i + 1 < chars.len() && chars[i + 1] == 'ا' {
                ranges.push((i, i + 2));
                i += 2;
            } else {
                ranges.push((i, i + 1));
                i += 1;
            }
        }
    }
    ranges
}

/// Lay out one word: run the model per cluster and chain origins.
/// Returns clusters with pixel origins (y-down, baseline y = 0).
pub fn layout_word(font: &FieldFont, word: &str) -> Vec<Cluster> {
    let chars: Vec<char> = word.chars().collect();
    let scale = font.canvas.em_px / font.canvas.upm;
    let mut out = Vec::new();
    let mut ox_u = 0.0f64;
    let mut oy_u = 0.0f64;
    for (ci, (letters, feats)) in word_clusters(font, &chars).into_iter().enumerate() {
        let g = font.glyph(feats);
        if ci == 0 {
            // first cluster's "displacement" is its absolute origin
            ox_u = g.ddx;
            oy_u = g.ddy;
        } else {
            ox_u += g.ddx;
            oy_u += g.ddy;
        }
        out.push(Cluster {
            letters,
            feats,
            ox: ox_u * scale,
            oy: -oy_u * scale, // font units are y-up; pixels are y-down
        });
    }
    out
}

/// A composited word: an SDF grid in pixels, plus its placement
/// relative to the word's baseline origin.
pub struct WordField {
    pub grid: Vec<f32>,
    pub w: usize,
    pub h: usize,
    /// Word-space position of the grid's top-left (y-down px,
    /// baseline at y = 0).
    pub x0: f64,
    pub y0: f64,
    pub clusters: Vec<Cluster>,
}

pub fn compose_word(font: &FieldFont, word: &str) -> WordField {
    let clusters = if word.contains(' ') {
        layout_text(font, word).into_iter().next().unwrap_or_default()
    } else {
        layout_word(font, word)
    };
    compose_clusters(font, clusters, None)
}

/// The feature tuple of a space between two words: the last two
/// letters before it and the first two after it.
pub fn space_feats(font: &FieldFont, before: &[char], after: &[char]) -> Option<[u32; 5]> {
    let sp = font.vocab_id(" ")?;
    let none = font.none_id();
    let id = |c: Option<&char>| -> u32 {
        c.and_then(|c| font.vocab_id(&c.to_string())).unwrap_or(none)
    };
    let n = before.len();
    Some([
        id(if n >= 2 { before.get(n - 2) } else { None }),
        id(before.last()),
        sp,
        id(after.first()),
        id(after.get(1)),
    ])
}

/// Lay out text that may contain spaces. Words joined by a trained
/// space (see `FieldFont::space_trained`) form one group in a shared
/// frame: the space is a cluster whose displacement carries the
/// next word's first origin, so a traced phrase keeps its
/// composition. An untrained space starts a new group, which the
/// caller places as it places words today. A space cluster has the
/// letters " " and is never drawn.
pub fn layout_text(font: &FieldFont, text: &str) -> Vec<Vec<Cluster>> {
    let scale = font.canvas.em_px / font.canvas.upm;
    let words: Vec<Vec<char>> =
        text.split(' ').filter(|w| !w.is_empty()).map(|w| w.chars().collect()).collect();
    let mut groups: Vec<Vec<Cluster>> = Vec::new();
    for (wi, chars) in words.iter().enumerate() {
        let word: String = chars.iter().collect();
        let mut cl = layout_word(font, &word);
        let chained = wi > 0
            && space_feats(font, &words[wi - 1], chars)
                .filter(|f| font.space_trained([f[0], f[1], f[3], f[4]]))
                .map(|f| {
                    let g = font.glyph(f);
                    let group = groups.last_mut().unwrap();
                    let last = group.last().unwrap();
                    // the space sits at the previous origin; the next
                    // word's first origin is that plus its displacement
                    let (sx, sy) = (last.ox, last.oy);
                    let (nx, ny) = (sx + g.ddx * scale, sy - g.ddy * scale);
                    group.push(Cluster { letters: " ".into(), feats: f, ox: sx, oy: sy });
                    if let Some(first) = cl.first() {
                        let (dx, dy) = (nx - first.ox, ny - first.oy);
                        for c in cl.iter_mut() {
                            c.ox += dx;
                            c.oy += dy;
                        }
                    }
                    group.extend(cl.drain(..));
                })
                .is_some();
        if !chained {
            groups.push(cl);
        }
    }
    groups
}

/// Compose a word's clusters into one field. `mask` selects a subset
/// by cluster index (None = all). The grid and its placement cover
/// only the included clusters; `clusters` keeps the full list.
pub fn compose_clusters(
    font: &FieldFont,
    clusters: Vec<Cluster>,
    mask: Option<&[bool]>,
) -> WordField {
    let included = |k: usize| {
        clusters[k].letters != " " && mask.map_or(true, |m| m.get(k).copied().unwrap_or(false))
    };
    let (cw, ch) = (font.canvas.w as f64, font.canvas.h as f64);
    let (cox, coy) = (font.canvas.origin_x, font.canvas.origin_y);
    // extents
    let mut x0 = f64::MAX;
    let mut y0 = f64::MAX;
    let mut x1 = f64::MIN;
    let mut y1 = f64::MIN;
    let mut n_inc = 0usize;
    for (k, c) in clusters.iter().enumerate() {
        if !included(k) {
            continue;
        }
        n_inc += 1;
        x0 = x0.min(c.ox - cox);
        y0 = y0.min(c.oy - coy);
        x1 = x1.max(c.ox - cox + cw);
        y1 = y1.max(c.oy - coy + ch);
    }
    if n_inc == 0 {
        return WordField { grid: vec![], w: 0, h: 0, x0: 0.0, y0: 0.0, clusters };
    }
    let w = (x1 - x0).ceil() as usize + 1;
    let h = (y1 - y0).ceil() as usize + 1;
    let mut grid = vec![-1.0f32; w * h];
    for (k, c) in clusters.iter().enumerate() {
        if !included(k) {
            continue;
        }
        let g = font.glyph(c.feats);
        let bx = (c.ox - cox - x0).round() as i64;
        let by = (c.oy - coy - y0).round() as i64;
        for y in 0..font.canvas.h {
            let ty = by + y as i64;
            if ty < 0 || ty as usize >= h {
                continue;
            }
            for x in 0..font.canvas.w {
                let tx = bx + x as i64;
                if tx < 0 || tx as usize >= w {
                    continue;
                }
                let v = g.field[y * font.canvas.w + x];
                let cell = &mut grid[ty as usize * w + tx as usize];
                if v > *cell {
                    *cell = v;
                }
            }
        }
    }
    WordField { grid, w, h, x0, y0, clusters }
}

/// Trace with sub-pixel smoothing: bilinearly upsample the field 2x,
/// run marching squares on the finer grid, fit smooth curves to the
/// traced polygons, and scale back. The SDF is a continuous surface,
/// so interpolation recovers detail the pixel-rate trace drops; the
/// curve fit removes the stair steps.
pub fn trace_field_smooth(grid: &[f32], w: usize, h: usize) -> BezPath {
    // 3x upsample and a fit tolerance of 1.2 upsampled pixels (0.4
    // source pixels): chosen visually against 2x/0.8 and 4x/1.6 on
    // نستعليق renders; 4x is marginally rounder at twice the cost.
    trace_field_smooth_with(grid, w, h, 3, 1.2)
}

/// Parameterized smooth trace: `s` is the upsample factor, `accuracy`
/// is the curve-fit tolerance in upsampled pixels.
pub fn trace_field_smooth_with(
    grid: &[f32],
    w: usize,
    h: usize,
    s: usize,
    accuracy: f64,
) -> BezPath {
    if w == 0 || h == 0 {
        return BezPath::new();
    }
    let (uw, uh) = (w * s, h * s);
    let mut up = vec![-1.0f32; uw * uh];
    for y in 0..uh {
        let fy = ((y as f32 + 0.5) / s as f32 - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(h - 1);
        let y1 = (y0 + 1).min(h - 1);
        let ty = (fy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..uw {
            let fx = ((x as f32 + 0.5) / s as f32 - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(w - 1);
            let x1 = (x0 + 1).min(w - 1);
            let tx = (fx - x0 as f32).clamp(0.0, 1.0);
            let a = grid[y0 * w + x0] * (1.0 - tx) + grid[y0 * w + x1] * tx;
            let b = grid[y1 * w + x0] * (1.0 - tx) + grid[y1 * w + x1] * tx;
            up[y * uw + x] = a * (1.0 - ty) + b * ty;
        }
    }
    let traced = trace_field(&up, uw, uh);
    let opts = kurbo::simplify::SimplifyOptions::default();
    let fitted = kurbo::simplify::simplify_bezpath(traced, accuracy, &opts);
    kurbo::Affine::scale(1.0 / s as f64) * fitted
}

/// Marching squares with linear interpolation: extract the iso-0
/// contour of an SDF grid as closed polyline paths, sub-pixel
/// accurate. Coordinates are in grid pixels.
pub fn trace_field(grid: &[f32], w: usize, h: usize) -> BezPath {
    let f = |x: usize, y: usize| grid[y * w + x];
    let lerp = |a: f32, b: f32| a as f64 / (a - b) as f64;
    let key = |p: (f64, f64)| ((p.0 * 1024.0).round() as i64, (p.1 * 1024.0).round() as i64);

    let mut segs: HashMap<(i64, i64), Vec<((f64, f64), (f64, f64))>> = HashMap::new();
    let mut add = |a: (f64, f64), b: (f64, f64)| {
        segs.entry(key(a)).or_default().push((a, b));
    };
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let (tl, tr, bl, br) = (f(x, y), f(x + 1, y), f(x, y + 1), f(x + 1, y + 1));
            let case = ((tl >= 0.0) as u8)
                | (((tr >= 0.0) as u8) << 1
                    | ((br >= 0.0) as u8) << 2
                    | ((bl >= 0.0) as u8) << 3);
            if case == 0 || case == 15 {
                continue;
            }
            let xf = x as f64;
            let yf = y as f64;
            let top = (xf + lerp(tl, tr), yf);
            let bottom = (xf + lerp(bl, br), yf + 1.0);
            let left = (xf, yf + lerp(tl, bl));
            let right = (xf + 1.0, yf + lerp(tr, br));
            // Segments oriented with inside (>=0) on the left.
            match case {
                1 => add(top, left),
                2 => add(right, top),
                3 => add(right, left),
                4 => add(bottom, right),
                5 => {
                    // ambiguous saddle: split by center sign
                    if (tl + tr + bl + br) >= 0.0 {
                        add(top, right);
                        add(bottom, left);
                    } else {
                        add(top, left);
                        add(bottom, right);
                    }
                }
                6 => add(bottom, top),
                7 => add(bottom, left),
                8 => add(left, bottom),
                9 => add(top, bottom),
                10 => {
                    if (tl + tr + bl + br) >= 0.0 {
                        add(right, bottom);
                        add(left, top);
                    } else {
                        add(right, top);
                        add(left, bottom);
                    }
                }
                11 => add(right, bottom),
                12 => add(left, right),
                13 => add(top, right),
                14 => add(left, top),
                _ => {}
            }
        }
    }

    let mut path = BezPath::new();
    loop {
        let Some((&start_key, _)) = segs.iter().next() else { break };
        let mut list = segs.remove(&start_key).unwrap();
        let (start, mut cur) = list.pop().unwrap();
        if !list.is_empty() {
            segs.insert(start_key, list);
        }
        path.move_to((start.0, start.1));
        let mut steps = 0;
        while key(cur) != key(start) && steps < 1_000_000 {
            path.line_to((cur.0, cur.1));
            let k = key(cur);
            let Some(nexts) = segs.get_mut(&k) else { break };
            let (_, nxt) = nexts.pop().unwrap();
            if nexts.is_empty() {
                segs.remove(&k);
            }
            cur = nxt;
            steps += 1;
        }
        path.close_path();
    }
    path
}

/// Compose a word with some clusters pulled away from their default
/// places. `pulls[k]` moves cluster `k`, and every cluster after it,
/// by that much (font units, y up) relative to cluster `k - 1`. Where
/// the two clusters join, the ink between them stretches to follow
/// (see `stretch`); where they do not touch, the cluster just moves.
pub fn compose_word_pulled(font: &FieldFont, word: &str, pulls: &[(f64, f64)]) -> WordField {
    let scale = font.canvas.em_px / font.canvas.upm;
    let px: Vec<(f64, f64)> = pulls.iter().map(|&(dx, dy)| (dx * scale, -dy * scale)).collect();
    compose_clusters_pulled(font, layout_word(font, word), &px)
}

/// `compose_word_pulled` on clusters already laid out, with the pulls
/// in field pixels (y down), as a dragged node reports them.
///
/// A font trained for it draws the pulled fields itself, with the
/// pulls as inputs to the network. Any other font gets the geometric
/// stretch applied to its default fields.
pub fn compose_clusters_pulled(
    font: &FieldFont,
    clusters: Vec<Cluster>,
    pulls: &[(f64, f64)],
) -> WordField {
    compose_clusters_pulled_with(font, clusters, pulls, font.learned_stretch())
}

/// As `compose_clusters_pulled`, choosing the learned or the geometric
/// stretch explicitly (for comparing the two).
pub fn compose_clusters_pulled_with(
    font: &FieldFont,
    mut clusters: Vec<Cluster>,
    pulls: &[(f64, f64)],
    learned: bool,
) -> WordField {
    use crate::stretch::{applied, pulled, Geometry, Neighbor};
    let g = Geometry {
        w: font.canvas.w,
        h: font.canvas.h,
        em_px: font.canvas.em_px as f32,
        spread_px: font.canvas.spread_px as f32,
    };
    let n = clusters.len();
    // Fields composite at whole pixels. Two clusters draw the same
    // stretched stroke, each in its own frame, so both must agree on
    // where they sit to the pixel: snap the origins and the pulls.
    let (cox, coy) = (font.canvas.origin_x, font.canvas.origin_y);
    for c in clusters.iter_mut() {
        c.ox = (c.ox - cox).round() + cox;
        c.oy = (c.oy - coy).round() + coy;
    }
    let pull = |k: usize| -> (f32, f32) {
        let (dx, dy) = pulls.get(k).copied().unwrap_or((0.0, 0.0));
        (dx.round() as f32, dy.round() as f32)
    };
    let base: Vec<std::rc::Rc<crate::field_model::GlyphField>> =
        clusters.iter().map(|c| font.glyph(c.feats)).collect();
    let offset = |from: usize, to: usize| -> (i64, i64) {
        (
            (clusters[to].ox - clusters[from].ox).round() as i64,
            (clusters[to].oy - clusters[from].oy).round() as i64,
        )
    };
    let mut fields: Vec<Vec<f32>> = Vec::with_capacity(n);
    // the pull each join really got (pushes are clamped)
    let mut moved: Vec<(f32, f32)> = vec![(0.0, 0.0); n];
    for k in 0..n {
        if clusters[k].letters == " " {
            fields.push(base[k].field.clone());
            moved[k] = pull(k);
            continue;
        }
        let prev_nb = (k > 0 && clusters[k - 1].letters != " ").then(|| {
            let (dx, dy) = offset(k, k - 1);
            Neighbor { field: &base[k - 1].field, dx, dy }
        });
        let next_nb = (k + 1 < n && clusters[k + 1].letters != " ").then(|| {
            let (dx, dy) = offset(k, k + 1);
            Neighbor { field: &base[k + 1].field, dx, dy }
        });
        let prev = prev_nb.as_ref().map(|nb| (nb, pull(k)));
        let next = next_nb.as_ref().map(|nb| (nb, pull(k + 1)));
        let (field, done) = if learned {
            // A font trained on drawn stretch learned its pulls from
            // drawings, which meet edge to edge rather than overlapping
            // the way an OpenType font's joins do: every letter with a
            // neighbor takes the pull, as in training, without looking
            // for shared ink first.
            let clamp = |d: (f32, f32)| (d.0.min(g.max_push()), d.1);
            let done = crate::stretch::Applied {
                prev: prev.map(|(_, d)| clamp(d)),
                next: next.map(|(_, d)| clamp(d)),
            };
            (font.glyph_pulled(clusters[k].feats, done.cond(g.em_px)).field.clone(), done)
        } else {
            pulled(&base[k].field, &g, prev, next)
        };
        fields.push(field);
        moved[k] = done.prev.unwrap_or(pull(k));
    }
    // Moved origins: each pull carries the rest of the word with it.
    let (mut sx, mut sy) = (0.0f64, 0.0f64);
    for (k, c) in clusters.iter_mut().enumerate() {
        sx += moved[k].0 as f64;
        sy += moved[k].1 as f64;
        c.ox += sx;
        c.oy += sy;
    }
    compose_fields(font, clusters, &fields)
}

/// Composite explicit per-cluster fields at the clusters' origins.
pub fn compose_fields(font: &FieldFont, clusters: Vec<Cluster>, fields: &[Vec<f32>]) -> WordField {
    let (cw, ch) = (font.canvas.w, font.canvas.h);
    let (cox, coy) = (font.canvas.origin_x, font.canvas.origin_y);
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for c in clusters.iter().filter(|c| c.letters != " ") {
        x0 = x0.min(c.ox - cox);
        y0 = y0.min(c.oy - coy);
        x1 = x1.max(c.ox - cox + cw as f64);
        y1 = y1.max(c.oy - coy + ch as f64);
    }
    if x0 == f64::MAX {
        return WordField { grid: vec![], w: 0, h: 0, x0: 0.0, y0: 0.0, clusters };
    }
    let w = (x1 - x0).ceil() as usize + 1;
    let h = (y1 - y0).ceil() as usize + 1;
    let mut grid = vec![-1.0f32; w * h];
    for (c, field) in clusters.iter().zip(fields).filter(|(c, _)| c.letters != " ") {
        let bx = (c.ox - cox - x0).round() as i64;
        let by = (c.oy - coy - y0).round() as i64;
        for y in 0..ch {
            let ty = by + y as i64;
            if ty < 0 || ty as usize >= h {
                continue;
            }
            for x in 0..cw {
                let tx = bx + x as i64;
                if tx < 0 || tx as usize >= w {
                    continue;
                }
                let cell = &mut grid[ty as usize * w + tx as usize];
                *cell = cell.max(field[y * cw + x]);
            }
        }
    }
    WordField { grid, w, h, x0, y0, clusters }
}
