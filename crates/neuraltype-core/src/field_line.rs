//! One line of text in a field font, laid out as the web demo and the
//! editor both show it: words placed right to left on one baseline, a
//! dragged node's pull applied to its letter, and the marks a viewer
//! draws over the ink -- one node per caret index and one span per
//! character. Coordinates are field pixels, y down.
//!
//! Both viewers call this, so a change to layout or to where a node
//! sits shows up in both at once.

use crate::field_model::FieldFont;
use crate::field_text;

/// Layout for field fonts: words composed by the model, laid out RTL
/// on a shared baseline. Coordinates are in field pixels (em_px per
/// em); the JSON contract matches the v0 shape() so the demo island
/// works unchanged. Extra field-font data: `nodes`, one 2D point per
/// logical caret index, on the displacement chain.
pub struct PlacedWord {
    pub wf: field_text::WordField,
    pub dx: f64,
    pub char_base: usize,
    pub n_chars: usize,
    /// Ink edges in chain coordinates, measured before drag offsets.
    pub ink_l: f64,
    pub ink_r: f64,
    /// Chain-space y of the word's ink at its left (exit) edge: the
    /// end-of-word caret node sits here, on the tail of the last
    /// letter instead of floating mid-air.
    pub exit_y: f64,
    /// Chain-space y of the word's ink at its right (entry) edge:
    /// the before-the-word caret node sits here, on the first
    /// letter's entry stroke.
    pub entry_y: f64,
}

pub struct FieldLine {
    pub words: Vec<PlacedWord>,
    pub width: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub space: f64,
    pub total_chars: usize,
}

pub type NodeOffsets = std::collections::HashMap<usize, (f64, f64)>;

pub fn build_field_line(f: &FieldFont, text: &str, offsets: &NodeOffsets) -> FieldLine {
    let em = f.canvas.em_px;
    // Gulzar's own space glyph advances 0.12 em (hmtx); match it.
    let space = 0.12 * em;
    let mut pen_right = 0.0f64;
    let mut y_min = -1.2 * em;
    let mut y_max = 0.5 * em;
    let mut words = Vec::new();
    let mut char_base = 0usize;
    for word in text.split(' ') {
        let n_chars = word.chars().count();
        if word.is_empty() {
            pen_right -= space;
            char_base += 1;
            continue;
        }
        // layout, then apply dragged-node offsets. An offset at node
        // i belongs to the cluster that ENDS at i, which is the
        // letter the caret hint outlines (the caret at i sits after
        // letter i-1). Placement anchors on the UNOFFSET ink, so a
        // drag moves the letter on screen instead of being cancelled
        // by the pen re-anchoring.
        let clusters = field_text::layout_word(f, word);
        let mut ci = 0usize;
        let mut has_off = false;
        for c in &clusters {
            ci += c.letters.chars().count();
            if offsets.contains_key(&(char_base + ci)) {
                has_off = true;
            }
        }
        // total sideways pull in this word: the words after it move
        // over by the same amount, so a stretched word makes room
        let mut pulled_x = 0.0f64;
        let base_wf = field_text::compose_clusters(f, clusters.clone(), None);
        if base_wf.w == 0 {
            char_base += n_chars + 1;
            continue;
        }
        let ink = |wf: &field_text::WordField| {
            let (mut ix0, mut ix1, mut iy0, mut iy1) =
                (usize::MAX, 0usize, usize::MAX, 0usize);
            for y in 0..wf.h {
                for x in 0..wf.w {
                    if wf.grid[y * wf.w + x] >= 0.0 {
                        ix0 = ix0.min(x);
                        ix1 = ix1.max(x);
                        iy0 = iy0.min(y);
                        iy1 = iy1.max(y);
                    }
                }
            }
            (ix0, ix1, iy0, iy1)
        };
        let (bx0, bx1, _, _) = ink(&base_wf);
        if bx0 == usize::MAX {
            char_base += n_chars + 1;
            continue;
        }
        // ink edges in chain coordinates, from the unoffset word
        let ink_r = base_wf.x0 + bx1 as f64 + 1.0;
        let ink_l = base_wf.x0 + bx0 as f64;
        // ink exit height: centroid of the leftmost few ink columns
        let mut ysum = 0.0f64;
        let mut yn = 0usize;
        for y in 0..base_wf.h {
            for x in bx0..(bx0 + 4).min(base_wf.w) {
                if base_wf.grid[y * base_wf.w + x] >= 0.0 {
                    ysum += y as f64;
                    yn += 1;
                }
            }
        }
        let exit_y = base_wf.y0 + if yn > 0 { ysum / yn as f64 } else { 0.0 };
        let mut ysum_r = 0.0f64;
        let mut yn_r = 0usize;
        for y in 0..base_wf.h {
            for x in bx1.saturating_sub(3)..=bx1 {
                if base_wf.grid[y * base_wf.w + x] >= 0.0 {
                    ysum_r += y as f64;
                    yn_r += 1;
                }
            }
        }
        let entry_y = base_wf.y0 + if yn_r > 0 { ysum_r / yn_r as f64 } else { 0.0 };
        let wf = if has_off {
            // A dragged node pulls its letter and the rest of the
            // word after it; the join before it stretches to follow.
            let mut pulls = vec![(0.0, 0.0); clusters.len()];
            let mut ci = 0usize;
            for (k, c) in clusters.iter().enumerate() {
                ci += c.letters.chars().count();
                if let Some(&off) = offsets.get(&(char_base + ci)) {
                    pulls[k] = off;
                    pulled_x += off.0;
                }
            }
            field_text::compose_clusters_pulled(f, clusters, &pulls)
        } else {
            base_wf
        };
        if wf.w == 0 {
            char_base += n_chars + 1;
            continue;
        }
        let (_, _, ry0, ry1) = ink(&wf);
        // place word: right edge of its unoffset INK at pen_right
        let dx = pen_right - ink_r;
        if ry0 != usize::MAX {
            y_min = y_min.min(wf.y0 + ry0 as f64);
            y_max = y_max.max(wf.y0 + ry1 as f64 + 1.0);
        }
        pen_right -= (ink_r - ink_l) + space - pulled_x.min(0.0);
        words.push(PlacedWord {
            wf,
            dx,
            char_base,
            n_chars,
            ink_l,
            ink_r,
            exit_y,
            entry_y,
        });
        char_base += n_chars + 1;
    }
    let width = -pen_right - space.min(-pen_right);
    FieldLine {
        words,
        width,
        y_min,
        y_max,
        space,
        total_chars: text.chars().count(),
    }
}

/// One character's horizontal extent on the line, for carets and hits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    pub i: usize,
    pub x: f64,
    pub w: f64,
}

/// The marks over a laid-out line: `nodes[i]` is the node at caret
/// index i (0 ..= the text's character count), `spans` one per
/// character in layout order.
pub struct Marks {
    pub nodes: Vec<(f64, f64)>,
    pub spans: Vec<Span>,
}

/// The nodes and spans of `line`, moved by `shift` across and by
/// `-y_min` down. The web demo passes the line's width and top, so its
/// coordinates start at the line's top right; pass 0 and 0 for the
/// chain's own frame, with the first word's right edge at x = 0 and
/// the baseline at y = 0.
pub fn marks(f: &FieldFont, line: &FieldLine, shift: f64, y_min: f64) -> Marks {
    let baseline_y = -y_min;
    let n = line.total_chars;
    let mut spans: Vec<Span> = Vec::new();
    let mut nodes: Vec<(f64, f64)> = vec![(f64::NAN, f64::NAN); n + 1];
    for pw in &line.words {
        let wf = &pw.wf;
        // visual join point between two clusters: the deepest cell of
        // their fields' intersection -- the middle of the connecting
        // stroke, which is where an insertion visually breaks the
        // word. None when the letters do not touch.
        let join_point = |ka: &field_text::Cluster,
                          kb: &field_text::Cluster|
         -> Option<(f64, f64)> {
            let (cw_i, ch_i) = (f.canvas.w as i64, f.canvas.h as i64);
            let (cox, coy) = (f.canvas.origin_x, f.canvas.origin_y);
            let ga = f.glyph(ka.feats);
            let gb = f.glyph(kb.feats);
            let ax0 = (ka.ox - cox).round() as i64;
            let ay0 = (ka.oy - coy).round() as i64;
            let bx0 = (kb.ox - cox).round() as i64;
            let by0 = (kb.oy - coy).round() as i64;
            let x0 = ax0.max(bx0);
            let y0 = ay0.max(by0);
            let x1 = (ax0 + cw_i).min(bx0 + cw_i);
            let y1 = (ay0 + ch_i).min(by0 + ch_i);
            let mut best = f32::MIN;
            let mut bp = None;
            for y in y0..y1 {
                for x in x0..x1 {
                    let va = ga.field[((y - ay0) * cw_i + (x - ax0)) as usize];
                    let vb = gb.field[((y - by0) * cw_i + (x - bx0)) as usize];
                    let m = va.min(vb);
                    if m > best {
                        best = m;
                        bp = Some((x as f64 + 0.5, y as f64 + 0.5));
                    }
                }
            }
            if best >= 0.0 {
                bp
            } else {
                None
            }
        };

        // the word's strokes, without its dots: nodes snap only to these
        let body = body_cells(wf, f.canvas.em_px);
        let pen_right_word = pw.dx + pw.ink_r + shift;
        let left_ink = pw.ink_l + pw.dx + shift;
        let cl = &wf.clusters;
        let mut ci = 0usize;
        for (k, c) in cl.iter().enumerate() {
            let right = if k == 0 { pen_right_word } else { cl[k - 1].ox + pw.dx + shift };
            let left = if k + 1 < cl.len() { cl[k + 1].ox + pw.dx + shift } else { left_ink };
            let nch = c.letters.chars().count();
            let cw_ = (right - left).max(1.0) / nch as f64;
            let ox = c.ox + pw.dx + shift;
            let oy = c.oy - y_min;
            for j in 0..nch {
                let i = pw.char_base + ci + j;
                spans.push(Span { i, x: right - (j as f64 + 1.0) * cw_, w: cw_ });
                if i <= n {
                    // Every node sits on the ink, in the middle of
                    // the stroke: a node is a handle on the writing.
                    let on_ink = |x: f64, y: f64| {
                        let (ix, iy) = snap_to_ink(wf, &body, x - pw.dx - shift, y + y_min);
                        (ix + pw.dx + shift, iy - y_min)
                    };
                    if k == 0 && j == 0 {
                        // before the word: where the first letter's
                        // ink begins
                        nodes[i] = on_ink(pw.ink_r - 1.0 + pw.dx + shift, pw.entry_y - y_min);
                    } else if nch == 1 {
                        // between letters: where their ink meets; where
                        // it does not, the ink nearest the next
                        // letter's start
                        nodes[i] = match join_point(&cl[k - 1], c) {
                            Some((jx, jy)) => (jx + pw.dx + shift, jy - y_min),
                            None => match nearest_between(f, &cl[k - 1], c) {
                                Some((ex, ey)) => on_ink(ex + pw.dx + shift, ey - y_min),
                                None => on_ink(ox, oy),
                            },
                        };
                    } else {
                        // ligatures: one node per character, spread
                        // across the cluster's visual span
                        nodes[i] = on_ink(right - (j as f64 + 0.5) * cw_, oy);
                    }
                }
            }
            ci += nch;
        }
        // after the word: where the last letter's ink ends
        let end_i = pw.char_base + pw.n_chars;
        if end_i <= n {
            let (ix, iy) = snap_to_ink(wf, &body, pw.ink_l + 1.0, pw.exit_y);
            nodes[end_i] = (ix + pw.dx + shift, iy - y_min);
        }
    }
    // fill gaps (leading/trailing spaces, unrendered words):
    // forward then backward copy of the nearest known node
    let mut last: Option<(f64, f64)> = None;
    for p in nodes.iter_mut() {
        if p.0.is_nan() {
            if let Some(q) = last {
                *p = q;
            }
        } else {
            last = Some(*p);
        }
    }
    let mut next: Option<(f64, f64)> = None;
    for p in nodes.iter_mut().rev() {
        if p.0.is_nan() {
            *p = next.unwrap_or((shift, baseline_y));
        } else {
            next = Some(*p);
        }
    }
    Marks { nodes, spans }
}

/// The outline around the letters from caret index `start` to `end`:
/// the union of their fields, traced at a raised level, so it hugs the
/// ink like the cloud bands around manuscript text. One path per word
/// that has selected letters, moved like `marks`. A one-letter range
/// is the hint around the letter before the caret.
pub fn selection_paths(
    f: &FieldFont,
    line: &FieldLine,
    start: usize,
    end: usize,
    shift: f64,
    y_min: f64,
) -> Vec<kurbo::BezPath> {
    let mut paths = Vec::new();
    if end <= start {
        return paths;
    }
    for pw in &line.words {
        let a = start.max(pw.char_base);
        let b = end.min(pw.char_base + pw.n_chars);
        if a >= b {
            continue;
        }
        let mut ci = 0usize;
        let mask: Vec<bool> = pw
            .wf
            .clusters
            .iter()
            .map(|c| {
                let nch = c.letters.chars().count();
                let cs = pw.char_base + ci;
                ci += nch;
                cs < end && cs + nch > start
            })
            .collect();
        let sel = field_text::compose_clusters(f, pw.wf.clusters.clone(), Some(&mask));
        if sel.w == 0 {
            continue;
        }
        // +0.45 of the spread (8 px) dilates the zero contour by
        // about 3.6 px
        let dil: Vec<f32> = sel.grid.iter().map(|v| v + 0.45).collect();
        let path = field_text::trace_field_smooth(&dil, sel.w, sel.h);
        paths.push(kurbo::Affine::translate((sel.x0 + pw.dx + shift, sel.y0 - y_min)) * path);
    }
    paths
}

/// A node is a gap when an edit there touches a word boundary: the
/// start or end of the text, or beside a space. Viewers draw it hollow.
pub fn is_gap(chars: &[char], i: usize) -> bool {
    i == 0 || i >= chars.len() || chars[i - 1] == ' ' || chars[i] == ' '
}

/// The strand: a natural cubic spline through the nodes in caret
/// order. C2 continuous (curvature never jumps), chord-length
/// parameterized so uneven node spacing does not kink the curve.
/// Coincident nodes (ligature interiors) collapse to one spline point
/// but keep their parameters. Same math as the web demo's buildStrand.
pub struct Strand {
    xs: Vec<f64>,
    ys: Vec<f64>,
    t: Vec<f64>,
    mx: Vec<f64>,
    my: Vec<f64>,
    t_of_index: Vec<usize>,
}

impl Strand {
    pub fn new(points: &[(f64, f64)]) -> Strand {
        let mut keep: Vec<usize> = Vec::new();
        let mut t_of_index = vec![0; points.len()];
        for (i, p) in points.iter().enumerate() {
            let far = keep.last().map_or(true, |&k| {
                let q = points[k];
                (p.0 - q.0).hypot(p.1 - q.1) > 0.75
            });
            if far {
                keep.push(i);
            }
            t_of_index[i] = keep.len().saturating_sub(1);
        }
        let xs: Vec<f64> = keep.iter().map(|&i| points[i].0).collect();
        let ys: Vec<f64> = keep.iter().map(|&i| points[i].1).collect();
        let mut t = vec![0.0];
        for k in 1..xs.len() {
            let chord = (xs[k] - xs[k - 1]).hypot(ys[k] - ys[k - 1]).max(1e-6);
            t.push(t[k - 1] + chord);
        }
        if xs.is_empty() {
            t.clear();
        }
        let mx = second_derivatives(&xs, &t);
        let my = second_derivatives(&ys, &t);
        Strand { xs, ys, t, mx, my, t_of_index }
    }

    /// The parameter at the end of the strand.
    pub fn t_end(&self) -> f64 {
        self.t.last().copied().unwrap_or(0.0)
    }

    /// The parameter of node `i`.
    pub fn t_of(&self, i: usize) -> f64 {
        if self.t_of_index.is_empty() {
            return 0.0;
        }
        self.t[self.t_of_index[i.min(self.t_of_index.len() - 1)]]
    }

    /// The point at parameter `u`.
    pub fn sample(&self, u: f64) -> (f64, f64) {
        let n = self.xs.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        if n == 1 {
            return (self.xs[0], self.ys[0]);
        }
        let u = u.clamp(self.t[0], self.t[n - 1]);
        let mut k = 0;
        while k < n - 2 && self.t[k + 1] < u {
            k += 1;
        }
        (
            eval_axis(&self.xs, &self.mx, &self.t, k, u),
            eval_axis(&self.ys, &self.my, &self.t, k, u),
        )
    }

    /// The strand from `u0` to `u1` as a polyline, a point every 3
    /// units or so, as the web demo strokes it.
    pub fn polyline(&self, u0: f64, u1: f64) -> Vec<(f64, f64)> {
        let steps = ((u1 - u0).abs() / 3.0).ceil().max(2.0) as usize;
        (0..=steps)
            .map(|k| self.sample(u0 + (u1 - u0) * k as f64 / steps as f64))
            .collect()
    }
}

/// Second derivatives of a natural spline through `v` at parameters
/// `t` (the Thomas algorithm).
fn second_derivatives(v: &[f64], t: &[f64]) -> Vec<f64> {
    let n = v.len();
    if n < 3 {
        return vec![0.0; n];
    }
    let (mut a, mut b, mut c, mut d) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    b[0] = 1.0;
    b[n - 1] = 1.0;
    for k in 1..n - 1 {
        let h0 = t[k] - t[k - 1];
        let h1 = t[k + 1] - t[k];
        a[k] = h0;
        b[k] = 2.0 * (h0 + h1);
        c[k] = h1;
        d[k] = 6.0 * ((v[k + 1] - v[k]) / h1 - (v[k] - v[k - 1]) / h0);
    }
    for k in 1..n {
        let m = a[k] / b[k - 1];
        b[k] -= m * c[k - 1];
        d[k] -= m * d[k - 1];
    }
    let mut m2 = vec![0.0; n];
    m2[n - 1] = d[n - 1] / b[n - 1];
    for k in (0..n - 1).rev() {
        m2[k] = (d[k] - c[k] * m2[k + 1]) / b[k];
    }
    m2
}

fn eval_axis(v: &[f64], m2: &[f64], t: &[f64], k: usize, u: f64) -> f64 {
    let h = t[k + 1] - t[k];
    let a = (t[k + 1] - u) / h;
    let b = (u - t[k]) / h;
    a * v[k] + b * v[k + 1] + ((a * a * a - a) * m2[k] + (b * b * b - b) * m2[k + 1]) * h * h / 6.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_strand_passes_through_its_nodes_and_merges_coincident_ones() {
        let nodes = [(0.0, 0.0), (10.0, 5.0), (10.2, 5.0), (30.0, 0.0), (40.0, -8.0)];
        let strand = Strand::new(&nodes);
        for (i, node) in nodes.iter().enumerate() {
            if i == 2 {
                // Within 0.75 of node 1: it shares node 1's parameter.
                assert_eq!(strand.t_of(2), strand.t_of(1));
                continue;
            }
            let p = strand.sample(strand.t_of(i));
            assert!((p.0 - node.0).abs() < 1e-9 && (p.1 - node.1).abs() < 1e-9, "{i}: {p:?}");
        }
        assert_eq!(strand.polyline(0.0, strand.t_end()).first(), Some(&(0.0, 0.0)));
    }

    #[test]
    fn gaps_are_the_ends_and_the_sides_of_spaces() {
        let chars: Vec<char> = "ab cd".chars().collect();
        let gaps: Vec<bool> = (0..=chars.len()).map(|i| is_gap(&chars, i)).collect();
        assert_eq!(gaps, vec![true, false, true, true, false, true]);
    }
}

/// The middle of the stroke nearest `(x, y)` in a word's own frame
/// (field pixels, y down, the word's chain origin): the nearest inked
/// cell, then the deepest cell close to it, so a node sits mid-stroke.
/// The point itself when the word has no ink within reach.
fn snap_to_ink(wf: &field_text::WordField, body: &[bool], x: f64, y: f64) -> (f64, f64) {
    let (w, h) = (wf.w as i64, wf.h as i64);
    if w == 0 || h == 0 {
        return (x, y);
    }
    let gx = (x - wf.x0 - 0.5).round() as i64;
    let gy = (y - wf.y0 - 0.5).round() as i64;
    let at = |cx: i64, cy: i64| -> f32 {
        if cx < 0 || cy < 0 || cx >= w || cy >= h || !body[(cy * w + cx) as usize] {
            -1.0
        } else {
            wf.grid[(cy * w + cx) as usize]
        }
    };
    // nearest inked cell, searching outward ring by ring
    let mut nearest = None;
    'rings: for r in 0..48i64 {
        let mut best: Option<(i64, (i64, i64))> = None;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx.abs() != r && dy.abs() != r {
                    continue;
                }
                let (cx, cy) = (gx + dx, gy + dy);
                if at(cx, cy) >= 0.0 {
                    let d = dx * dx + dy * dy;
                    if best.map_or(true, |(bd, _)| d < bd) {
                        best = Some((d, (cx, cy)));
                    }
                }
            }
        }
        if let Some((_, cell)) = best {
            nearest = Some(cell);
            break 'rings;
        }
    }
    let Some((nx, ny)) = nearest else {
        return (x, y);
    };
    // the deepest cell within a small reach: the middle of the stroke
    // half a stroke's width: enough to center across the stroke, too
    // little to slide along it toward a thicker part
    let reach = 3i64;
    let mut deep = (at(nx, ny), nx, ny);
    for dy in -reach..=reach {
        for dx in -reach..=reach {
            if dx * dx + dy * dy > reach * reach {
                continue;
            }
            let v = at(nx + dx, ny + dy);
            if v > deep.0 {
                deep = (v, nx + dx, ny + dy);
            }
        }
    }
    (wf.x0 + deep.1 as f64 + 0.5, wf.y0 + deep.2 as f64 + 0.5)
}

/// The strand through a line's nodes, as the pen moved: between two
/// nodes of one word it runs along the middle of the stroke that joins
/// them; between words, or where no ink joins them, it is a straight
/// segment. `nodes` are as `marks` gives them, moved by `shift` and
/// `-y_min`. Returns the strand's points and, for each node, the index
/// of its point.
pub fn strand(
    line: &FieldLine,
    nodes: &[(f64, f64)],
    shift: f64,
    y_min: f64,
) -> (Vec<(f64, f64)>, Vec<usize>) {
    let mut points: Vec<(f64, f64)> = Vec::new();
    let mut at_node = Vec::with_capacity(nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        if i > 0 {
            let word = line
                .words
                .iter()
                .find(|pw| pw.char_base <= i - 1 && i <= pw.char_base + pw.n_chars);
            let path = word.and_then(|pw| {
                let to_word = |p: (f64, f64)| (p.0 - pw.dx - shift, p.1 + y_min);
                let path = ink_path(&pw.wf, to_word(nodes[i - 1]), to_word(*node))?;
                Some(
                    path.into_iter()
                        .map(|(x, y)| (x + pw.dx + shift, y - y_min))
                        .collect::<Vec<_>>(),
                )
            });
            if let Some(path) = path {
                // the path's own ends are the nodes themselves
                points.extend(path.into_iter().skip(1));
                if let Some(last) = points.last_mut() {
                    *last = *node;
                }
            } else {
                points.push(*node);
            }
        } else {
            points.push(*node);
        }
        at_node.push(points.len() - 1);
    }
    (points, at_node)
}

/// The cheapest path through a word's ink from `a` to `b` (word frame),
/// favoring deep cells so it keeps to the middle of the stroke, then
/// smoothed. None when no ink joins them near the straight line.
fn ink_path(wf: &field_text::WordField, a: (f64, f64), b: (f64, f64)) -> Option<Vec<(f64, f64)>> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let (w, h) = (wf.w as i64, wf.h as i64);
    let cell = |p: (f64, f64)| ((p.0 - wf.x0 - 0.5).round() as i64, (p.1 - wf.y0 - 0.5).round() as i64);
    let (sa, sb) = (cell(a), cell(b));
    // search a window around the two ends
    let margin = 40;
    let x0 = (sa.0.min(sb.0) - margin).max(0);
    let y0 = (sa.1.min(sb.1) - margin).max(0);
    let x1 = (sa.0.max(sb.0) + margin).min(w - 1);
    let y1 = (sa.1.max(sb.1) + margin).min(h - 1);
    if x1 < x0 || y1 < y0 {
        return None;
    }
    let (ww, wh) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
    let inside = |c: (i64, i64)| c.0 >= x0 && c.0 <= x1 && c.1 >= y0 && c.1 <= y1;
    if !inside(sa) || !inside(sb) {
        return None;
    }
    let idx = |c: (i64, i64)| (c.1 - y0) as usize * ww + (c.0 - x0) as usize;
    let value = |c: (i64, i64)| wf.grid[(c.1 * w + c.0) as usize];
    let mut cost = vec![f64::INFINITY; ww * wh];
    let mut from = vec![usize::MAX; ww * wh];
    let mut heap = BinaryHeap::new();
    cost[idx(sa)] = 0.0;
    heap.push((Reverse((0.0f64 * 1000.0) as u64), sa));
    let goal = idx(sb);
    while let Some((Reverse(c), at)) = heap.pop() {
        let here = idx(at);
        if here == goal {
            break;
        }
        if (c as f64) / 1000.0 > cost[here] + 1e-6 {
            continue;
        }
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
            let next = (at.0 + dx, at.1 + dy);
            if !inside(next) {
                continue;
            }
            let v = value(next);
            // ink only, with the ends allowed a hair outside it
            if v < 0.0 && next != sb {
                continue;
            }
            let step = if dx != 0 && dy != 0 { std::f64::consts::SQRT_2 } else { 1.0 };
            let depth = (v.clamp(0.0, 1.0)) as f64;
            let total = cost[here] + step * (1.0 + 4.0 * (1.0 - depth));
            let n = idx(next);
            if total < cost[n] {
                cost[n] = total;
                from[n] = here;
                heap.push((Reverse((total * 1000.0) as u64), next));
            }
        }
    }
    if !cost[goal].is_finite() {
        return None;
    }
    let mut cells = Vec::new();
    let mut at = goal;
    while at != usize::MAX {
        cells.push(((at % ww) as i64 + x0, (at / ww) as i64 + y0));
        if at == idx(sa) {
            break;
        }
        at = from[at];
    }
    cells.reverse();
    let mut path: Vec<(f64, f64)> = cells
        .iter()
        .map(|c| (wf.x0 + c.0 as f64 + 0.5, wf.y0 + c.1 as f64 + 0.5))
        .collect();
    path[0] = a;
    let last = path.len() - 1;
    path[last] = b;
    // smooth the stair steps away, ends held
    for _ in 0..4 {
        let prev = path.clone();
        for k in 1..prev.len().saturating_sub(1) {
            path[k] = (
                (prev[k - 1].0 + 2.0 * prev[k].0 + prev[k + 1].0) / 4.0,
                (prev[k - 1].1 + 2.0 * prev[k].1 + prev[k + 1].1) / 4.0,
            );
        }
    }
    // every other point is plenty
    let mut thin: Vec<(f64, f64)> = path.iter().step_by(2).copied().collect();
    if thin.last() != path.last() {
        thin.push(*path.last().unwrap());
    }
    Some(thin)
}

/// Which cells of a word's field are stroke, not dot: inked cells in a
/// connected piece of at least a fifth of an em square, or longer than
/// 0.3 em, like a thin alif. Dots and other marks are small and compact.
fn body_cells(wf: &field_text::WordField, em_px: f64) -> Vec<bool> {
    let (w, h) = (wf.w, wf.h);
    let mut body = vec![false; w * h];
    let mut seen = vec![false; w * h];
    let least = ((0.2 * em_px) * (0.2 * em_px)).max(16.0) as usize;
    let long = (0.3 * em_px).max(4.0) as usize;
    let mut stack = Vec::new();
    for start in 0..w * h {
        if seen[start] || wf.grid[start] < 0.0 {
            continue;
        }
        let mut piece = Vec::new();
        let (mut lo_x, mut hi_x, mut lo_y, mut hi_y) = (usize::MAX, 0, usize::MAX, 0);
        seen[start] = true;
        stack.push(start);
        while let Some(at) = stack.pop() {
            piece.push(at);
            let (x, y) = (at % w, at / w);
            lo_x = lo_x.min(x);
            hi_x = hi_x.max(x);
            lo_y = lo_y.min(y);
            hi_y = hi_y.max(y);
            let mut visit = |n: usize| {
                if !seen[n] && wf.grid[n] >= 0.0 {
                    seen[n] = true;
                    stack.push(n);
                }
            };
            if x > 0 {
                visit(at - 1);
            }
            if x + 1 < w {
                visit(at + 1);
            }
            if y > 0 {
                visit(at - w);
            }
            if y + 1 < h {
                visit(at + w);
            }
        }
        if piece.len() >= least || (hi_x - lo_x).max(hi_y - lo_y) >= long {
            for at in piece {
                body[at] = true;
            }
        }
    }
    body
}

/// Where letter `a`'s ink comes closest to letter `b`'s, on `a`'s side
/// (word frame, field pixels): the exit toward a letter it does not
/// touch. None when either has no ink.
fn nearest_between(
    f: &FieldFont,
    a: &field_text::Cluster,
    b: &field_text::Cluster,
) -> Option<(f64, f64)> {
    let (cw, ch) = (f.canvas.w, f.canvas.h);
    let (cox, coy) = (f.canvas.origin_x, f.canvas.origin_y);
    let ink = |c: &field_text::Cluster| -> Vec<(f64, f64)> {
        let g = f.glyph(c.feats);
        let (x0, y0) = ((c.ox - cox).round(), (c.oy - coy).round());
        let mut cells = Vec::new();
        for y in (0..ch).step_by(2) {
            for x in (0..cw).step_by(2) {
                if g.field[y * cw + x] >= 0.0 {
                    cells.push((x0 + x as f64 + 0.5, y0 + y as f64 + 0.5));
                }
            }
        }
        cells
    };
    let (ia, ib) = (ink(a), ink(b));
    let mut best: Option<(f64, (f64, f64))> = None;
    for p in &ia {
        for q in &ib {
            let d = (p.0 - q.0).powi(2) + (p.1 - q.1).powi(2);
            if best.map_or(true, |(bd, _)| d < bd) {
                best = Some((d, *p));
            }
        }
    }
    best.map(|(_, p)| p)
}
