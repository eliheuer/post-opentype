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
                    if k == 0 && j == 0 {
                        // the before-the-word slot: on the first
                        // letter's ink entry, not the abstract pen
                        // origin
                        nodes[i] =
                            (pw.ink_r + pw.dx + shift + line.space * 0.25, pw.entry_y - y_min);
                    } else if nch == 1 {
                        // interior break: the visual junction where
                        // this letter's ink meets the previous
                        // letter's ink; chain origin when they do
                        // not touch
                        nodes[i] = match join_point(&cl[k - 1], c) {
                            Some((jx, jy)) => (jx + pw.dx + shift, jy - y_min),
                            None => (ox, oy),
                        };
                    } else {
                        // ligatures: one node per character cell,
                        // spread across the cluster's visual span --
                        // the origin can sit at either end of a wide
                        // ligature, and stacking nodes there clumps
                        // the strand
                        nodes[i] = (right - (j as f64 + 0.5) * cw_, oy);
                    }
                }
            }
            ci += nch;
        }
        // end-of-word caret: just past the left ink edge, at the
        // height where the word's ink actually exits
        let end_i = pw.char_base + pw.n_chars;
        if end_i <= n {
            nodes[end_i] = (left_ink - line.space * 0.5, pw.exit_y - y_min);
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
