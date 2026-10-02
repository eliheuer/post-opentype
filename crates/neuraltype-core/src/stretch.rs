//! Stretch: how a cluster's field deforms when a neighbor is moved
//! away from its default place.
//!
//! Two joined clusters share ink at their join. Moving the second by
//! `d` leaves each cluster's body where it is in its own frame and
//! deforms a narrow zone around the join: the zone is stretched along
//! x, sheared along y, and sags a little, like a kashida. Both
//! clusters apply the same global deformation, each in its own frame,
//! so their fields still meet when composited.
//!
//! The trainer uses this to make targets (the model learns to draw
//! the deformed field from the offsets as inputs), and the engine can
//! use it directly on fields from a font that was not trained for it.
//!
//! Text is right-to-left: a cluster's next neighbor is on its left
//! (smaller x), its previous neighbor on its right.

/// Half-width of the zone that deforms, in em.
pub const ZONE_EM: f32 = 0.02;
/// Sag at the middle of the stretched zone, per unit of stretch.
pub const SAG: f32 = 0.07;
/// How much the stretched stroke swells at its middle, in em of
/// extra half-thickness, reached at a stretch of `SWELL_AT_EM`. The
/// thin point of a join is a hairline; a kashida is a full stroke.
pub const SWELL_EM: f32 = 0.028;
pub const SWELL_AT_EM: f32 = 0.35;
/// Half-height of the band around the join that deforms, in em. Ink
/// above or below the band (dots, a stroke of the same letter that
/// overhangs the join) stays with its letter.
pub const BAND_EM: f32 = 0.09;
/// How far from the shared ink to look for the thin connecting
/// stroke, in em.
pub const REACH_EM: f32 = 0.22;

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A join and how far it is pulled. All values are canvas pixels with
/// y down. `at` is the join in the cluster's frame; `d` is how far the
/// neighbor on that side moved from its default place.
#[derive(Clone, Copy, Default)]
pub struct Pull {
    pub at: Option<(f32, f32)>,
    /// Half the stroke's thickness at the join, in pixels.
    pub half: f32,
    pub d: (f32, f32),
}

/// Deform one cluster field. `prev` is the join with the previous
/// cluster (on the right), where this cluster itself moved by
/// `prev.d`; `next` is the join with the next cluster (on the left),
/// which moved by `next.d`. `zone` is the zone half-width in pixels.
pub fn warp(
    field: &[f32],
    w: usize,
    h: usize,
    prev: Pull,
    next: Pull,
    zone: f32,
    band: f32,
    em_px: f32,
    spread_px: f32,
) -> Vec<f32> {
    let active = |p: &Pull| p.at.is_some() && (p.d.0 != 0.0 || p.d.1 != 0.0);
    if !active(&prev) && !active(&next) {
        return field.to_vec();
    }
    // weights: 0 in the body, 1 past the join
    let s_next = |x: f32| match next.at {
        Some((jx, _)) if active(&next) => smooth((jx + zone - x) / (2.0 * zone)),
        _ => 0.0,
    };
    let s_prev = |x: f32| match prev.at {
        Some((jx, _)) if active(&prev) => smooth((x - (jx - zone)) / (2.0 * zone)),
        _ => 0.0,
    };
    // forward map of a source column: where it lands, and its y shift
    let fx = |x: f32| x + next.d.0 * s_next(x) - prev.d.0 * s_prev(x);
    let fy = |x: f32| {
        let (sn, sp) = (s_next(x), s_prev(x));
        next.d.1 * sn - prev.d.1 * sp
            + SAG * next.d.0.abs() * 4.0 * sn * (1.0 - sn)
            + SAG * prev.d.0.abs() * 4.0 * sp * (1.0 - sp)
    };
    let reach = next.d.0.abs() + prev.d.0.abs() + 2.0;
    let mut out = vec![-1.0f32; w * h];
    for col in 0..w {
        // invert fx by bisection (fx is monotone for a stretch)
        let target = col as f32;
        let (mut lo, mut hi) = (target - reach, target + reach);
        for _ in 0..24 {
            let mid = 0.5 * (lo + hi);
            if fx(mid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let sx = 0.5 * (lo + hi);
        let dy = fy(sx);
        // swell: thickest mid-stroke, growing with the stretch
        let (sn, sp) = (s_next(sx), s_prev(sx));
        let grow = |d: f32| (d.abs() / (SWELL_AT_EM * em_px)).min(1.0);
        let swell = SWELL_EM * em_px / spread_px
            * (grow(next.d.0) * 4.0 * sn * (1.0 - sn) + grow(prev.d.0) * 4.0 * sp * (1.0 - sp));
        // which join this column belongs to, for the band
        let (band_y, half) = match (next.at, prev.at) {
            (Some(n), Some(p)) => {
                if s_next(sx) >= s_prev(sx) {
                    (n.1, next.half)
                } else {
                    (p.1, prev.half)
                }
            }
            (Some(n), None) => (n.1, next.half),
            (None, Some(p)) => (p.1, prev.half),
            (None, None) => (0.0, 0.0),
        };
        // The band hugs the stroke: other ink in these columns (a
        // tooth, a dot) must not be dragged along, even partly.
        let reach_y = (half + 1.5).min(band);
        for row in 0..h {
            // full strength inside the band, none outside it
            let v = 1.0 - smooth((row as f32 - dy - band_y).abs() - reach_y);
            let sx = target + (sx - target) * v;
            let sy = row as f32 - dy * v;
            if sx < 0.0 || sx > (w - 1) as f32 || sy < 0.0 || sy > (h - 1) as f32 {
                continue;
            }
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let tx = sx - x0 as f32;
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let ty = sy - y0 as f32;
            let a = field[y0 * w + x0] * (1.0 - tx) + field[y0 * w + x1] * tx;
            let b = field[y1 * w + x0] * (1.0 - tx) + field[y1 * w + x1] * tx;
            out[row * w + col] = (a * (1.0 - ty) + b * ty + swell * v).min(1.0);
        }
    }
    out
}

/// Where two consecutive clusters join, when the second sits at
/// `(dx, dy)` pixels from the first: the thinnest point of the stroke
/// that connects them, searched up to `reach` pixels either side of
/// the ink they share. Returned as (x, y, half thickness) in the
/// first cluster's frame; None when they do not touch.
pub fn join(
    a: &[f32],
    b: &[f32],
    w: usize,
    h: usize,
    dx: i64,
    dy: i64,
    reach: usize,
) -> Option<(f32, f32, f32)> {
    let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0usize);
    for y in 0..h as i64 {
        let by = y - dy;
        if by < 0 || by >= h as i64 {
            continue;
        }
        for x in 0..w as i64 {
            let bx = x - dx;
            if bx < 0 || bx >= w as i64 {
                continue;
            }
            if a[y as usize * w + x as usize] >= 0.0 && b[by as usize * w + bx as usize] >= 0.0 {
                sx += x as f64;
                sy += y as f64;
                n += 1;
            }
        }
    }
    if n == 0 {
        return None;
    }
    let (cx, cy) = ((sx / n as f64).round() as i64, (sy / n as f64).round() as i64);

    // The shared ink is often inside a thick part of a letter. A
    // stretch belongs on the thin stroke that connects the two, so
    // walk along the stroke both ways and take its thinnest column.
    let ink = |x: i64, y: i64| -> bool {
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            return false;
        }
        if a[y as usize * w + x as usize] >= 0.0 {
            return true;
        }
        let (bx, by) = (x - dx, y - dy);
        bx >= 0 && by >= 0 && bx < w as i64 && by < h as i64 && b[by as usize * w + bx as usize] >= 0.0
    };
    // the vertical run of ink in column x that overlaps rows [r0, r1]
    let run = |x: i64, r0: i64, r1: i64| -> Option<(i64, i64)> {
        let seed = (r0..=r1).find(|&y| ink(x, y))?;
        let (mut top, mut bot) = (seed, seed);
        while ink(x, top - 1) {
            top -= 1;
        }
        while ink(x, bot + 1) {
            bot += 1;
        }
        Some((top, bot))
    };
    let Some(start) = run(cx, cy, cy).or_else(|| run(cx, cy - 2, cy + 2)) else {
        return Some((cx as f32, cy as f32, 1.0));
    };
    let mut best = (start.1 - start.0, cx, start);
    for dir in [-1i64, 1] {
        let mut cur = start;
        for step in 1..=reach as i64 {
            let x = cx + dir * step;
            let Some(r) = run(x, cur.0, cur.1) else { break };
            if r.1 - r.0 < best.0 {
                best = (r.1 - r.0, x, r);
            }
            cur = r;
        }
    }
    Some((
        best.1 as f32,
        (best.2 .0 + best.2 .1) as f32 / 2.0,
        (best.2 .1 - best.2 .0 + 1) as f32 / 2.0,
    ))
}

/// A neighboring cluster: its field, and where its origin sits
/// relative to this cluster's origin at the default layout (whole
/// pixels, y down).
pub struct Neighbor<'a> {
    pub field: &'a [f32],
    pub dx: i64,
    pub dy: i64,
}

/// The canvas all fields share.
#[derive(Clone, Copy)]
pub struct Geometry {
    pub w: usize,
    pub h: usize,
    pub em_px: f32,
    pub spread_px: f32,
}

impl Geometry {
    fn zone(&self) -> f32 {
        ZONE_EM * self.em_px
    }
    /// The furthest a join can be pushed together (pixels).
    pub fn max_push(&self) -> f32 {
        self.zone() * 0.6
    }
}

/// What `pulled` did on each side: the pull it applied (whole pixels,
/// pushes clamped), or None where the clusters do not join.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Applied {
    pub prev: Option<(f32, f32)>,
    pub next: Option<(f32, f32)>,
}

/// The conditioning inputs are the pulls in em times this. The
/// weights that read them start at zero when a checkpoint is extended,
/// and the optimizer moves every weight at the same small rate; large
/// inputs let those few weights matter in hundreds of steps instead
/// of tens of thousands.
pub const COND_SCALE: f32 = 16.0;

impl Applied {
    /// The pulls as the model's conditioning input: [prev x, prev y,
    /// next x, next y] in em times `COND_SCALE`, zero where there is
    /// no join.
    pub fn cond(&self, em_px: f32) -> [f32; 4] {
        let p = self.prev.unwrap_or((0.0, 0.0));
        let n = self.next.unwrap_or((0.0, 0.0));
        let k = COND_SCALE / em_px;
        [p.0 * k, p.1 * k, n.0 * k, n.1 * k]
    }
}

/// Which sides join, and the pull each would get, without warping.
pub fn applied(
    field: &[f32],
    g: &Geometry,
    prev: Option<(&Neighbor, (f32, f32))>,
    next: Option<(&Neighbor, (f32, f32))>,
) -> Applied {
    let reach = (REACH_EM * g.em_px) as usize;
    let clamp = |d: (f32, f32)| (d.0.round().min(g.max_push()), d.1.round());
    Applied {
        prev: prev.and_then(|(nb, d)| {
            join(nb.field, field, g.w, g.h, -nb.dx, -nb.dy, reach).map(|_| clamp(d))
        }),
        next: next.and_then(|(nb, d)| {
            join(field, nb.field, g.w, g.h, nb.dx, nb.dy, reach).map(|_| clamp(d))
        }),
    }
}

/// One cluster's field after its neighbors are pulled: `prev.1` is
/// how far this cluster moved from its default place relative to the
/// previous one, `next.1` how far the next one moved relative to this
/// one (pixels, y down). The stretched stroke is drawn from the union
/// of the two clusters' ink, so both sides produce the same stroke.
pub fn pulled(
    field: &[f32],
    g: &Geometry,
    prev: Option<(&Neighbor, (f32, f32))>,
    next: Option<(&Neighbor, (f32, f32))>,
) -> (Vec<f32>, Applied) {
    let (w, h) = (g.w, g.h);
    let zone = g.zone();
    let band = BAND_EM * g.em_px;
    let reach = (REACH_EM * g.em_px) as usize;
    let clamp = |d: (f32, f32)| (d.0.round().min(g.max_push()), d.1.round());
    let mut src = field.to_vec();
    // union of both clusters' ink around the join at (jx, jy), in
    // this cluster's frame; the neighbor sits at (dx, dy)
    let mut unite = |src: &mut Vec<f32>, nb: &Neighbor, jx: f32, jy: f32| {
        let (x0, x1) = ((jx - zone - 2.0).floor() as i64, (jx + zone + 2.0).ceil() as i64);
        let (y0, y1) = ((jy - 1.5 * band).floor() as i64, (jy + 1.5 * band).ceil() as i64);
        for y in y0.max(0)..=y1.min(h as i64 - 1) {
            for x in x0.max(0)..=x1.min(w as i64 - 1) {
                let (bx, by) = (x - nb.dx, y - nb.dy);
                if bx < 0 || by < 0 || bx >= w as i64 || by >= h as i64 {
                    continue;
                }
                let i = y as usize * w + x as usize;
                src[i] = src[i].max(nb.field[by as usize * w + bx as usize]);
            }
        }
    };
    let mut done = Applied::default();
    let mut p_pull = Pull::default();
    if let Some((nb, d)) = prev {
        // the join is found in the previous cluster's frame
        if let Some((x, y, half)) = join(nb.field, field, w, h, -nb.dx, -nb.dy, reach) {
            let (jx, jy) = (x + nb.dx as f32, y + nb.dy as f32);
            unite(&mut src, nb, jx, jy);
            let d = clamp(d);
            p_pull = Pull { at: Some((jx, jy)), half, d };
            done.prev = Some(d);
        }
    }
    let mut n_pull = Pull::default();
    if let Some((nb, d)) = next {
        if let Some((jx, jy, half)) = join(field, nb.field, w, h, nb.dx, nb.dy, reach) {
            unite(&mut src, nb, jx, jy);
            let d = clamp(d);
            n_pull = Pull { at: Some((jx, jy)), half, d };
            done.next = Some(d);
        }
    }
    (warp(&src, w, h, p_pull, n_pull, zone, band, g.em_px, g.spread_px), done)
}
