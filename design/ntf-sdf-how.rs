// "How a signed-distance field works" figure for the Shapes as
// Fields section of the Nastaliq Distilled blog post.
//
// Left: the isolated خ as its real distance field (shape 6 of
// fields.bin, 155x219 u8, 128 = on the contour, spread 1/8 em =
// 8 px at 64 px/em), with a red box marking a small patch on the
// edge. Right: that patch blown up to a grid of cells, each cell
// printed with its signed distance in pixels, and the zero contour
// drawn through the patch at the interpolated sub-pixel positions.
use designbot::prelude::*;

const W_PX: usize = 155;
const H_PX: usize = 219;
const SHAPE: usize = 6; // isolated خ
// same crop window as ntf-sdf.rs (bbox x 58..103, y 48..130)
const CX: usize = 38;
const CY: usize = 28;
const CW: usize = 86;
const CH: usize = 122;
// zoom patch: N x N samples, chosen by the search in main()
const N: usize = 6;

fn main() {
    let bin = std::fs::read(concat!(
        env!("HOME"),
        "/GH/repos/post-opentype/data/fields-gulzar-64/fields.bin"
    ))
    .expect("fields.bin");
    let f = &bin[SHAPE * W_PX * H_PX..(SHAPE + 1) * W_PX * H_PX];
    let at = |x: usize, y: usize| f[y * W_PX + x];

    // Find a patch on the contour: mostly mid-range values (a live
    // gradient) with a balanced inside/outside split.
    let mut patch = (CX + 20, CY + 40);
    // skip the dot: search only the lower part of the crop, where
    // the main stroke of the bowl is
    'search: for y in CY + 60..CY + CH - N {
        for x in CX..CX + CW - N {
            let mut inside = 0;
            let mut near = 0;
            for j in 0..N {
                for i in 0..N {
                    let v = at(x + i, y + j);
                    if v >= 128 {
                        inside += 1;
                    }
                    if v.abs_diff(128) < 48 {
                        near += 1;
                    }
                }
            }
            if (14..=22).contains(&inside) && near >= 24 {
                patch = (x, y);
                break 'search;
            }
        }
    }
    let (px0, py0) = patch;

    const W: f64 = 2400.0;
    const H: f64 = 1260.0;
    let mut ctx = Canvas::new(W, H);
    ctx.background(Color::rgb(12, 12, 12));

    // ---- left panel: the field, drawn cell by cell (y-up canvas,
    // field rows are top-down, so flip) ----
    let scale = 8.0;
    let pw = CW as f64 * scale;
    let ph = CH as f64 * scale;
    let cell_l = scale;
    let gap = 120.0;
    let lx0 = gap;
    let ly0 = (H - ph) / 2.0;

    for cy in 0..CH {
        for cx in 0..CW {
            let (x, y) = (CX + cx, CY + cy);
            let v = at(x, y);
            // quantized gray bands, as in ntf-sdf.rs
            let q = (v / 32) as u32;
            let g = (25 + q * 27) as u8;
            ctx.no_stroke().fill(Color::rgb(g, g, g));
            let rx = lx0 + cx as f64 * cell_l;
            let ry = ly0 + (CH - 1 - cy) as f64 * cell_l;
            ctx.rect(rx, ry, cell_l, cell_l);
        }
    }
    ctx.no_fill().stroke(Color::rgb(70, 70, 70)).stroke_width(2.0);
    ctx.rect(lx0, ly0, pw, ph);

    // highlight box around the patch
    let hx = lx0 + (px0 - CX) as f64 * cell_l;
    let hy = ly0 + (CH as f64 - (py0 - CY + N) as f64) * cell_l;
    let hs = N as f64 * cell_l;
    ctx.no_fill().stroke(Color::rgb(239, 68, 68)).stroke_width(4.0);
    ctx.rect(hx, hy, hs, hs);

    // ---- right panel: the patch as a grid of numbered cells ----
    let cell = 150.0;
    let zs = N as f64 * cell;
    let zx0 = W - gap - zs;
    let zy0 = (H - zs) / 2.0;

    for j in 0..N {
        for i in 0..N {
            let v = at(px0 + i, py0 + j);
            let q = (v / 32) as u32;
            let g = (25 + q * 27) as u8;
            ctx.no_stroke().fill(Color::rgb(g, g, g));
            let rx = zx0 + i as f64 * cell;
            let ry = zy0 + (N - 1 - j) as f64 * cell;
            ctx.rect(rx, ry, cell, cell);
        }
    }
    // cell grid lines
    ctx.no_fill().stroke(Color::rgb(12, 12, 12)).stroke_width(3.0);
    for k in 1..N {
        let d = k as f64 * cell;
        ctx.line(zx0 + d, zy0, zx0 + d, zy0 + zs);
        ctx.line(zx0, zy0 + d, zx0 + zs, zy0 + d);
    }
    ctx.no_fill().stroke(Color::rgb(70, 70, 70)).stroke_width(2.0);
    ctx.rect(zx0, zy0, zs, zs);

    // signed distance value in each cell, in px of the 64 px/em grid
    ctx.font("IBM Plex Mono").font_size(44.0).text_align(TextAlign::Center);
    for j in 0..N {
        for i in 0..N {
            let v = at(px0 + i, py0 + j);
            let d = (v as f64 - 128.0) / 16.0;
            let q = (v / 32) as u32;
            let g = 25 + q * 27;
            let tc = if g > 120 {
                Color::rgb(12, 12, 12)
            } else {
                Color::rgb(200, 200, 200)
            };
            ctx.no_stroke().fill(tc);
            let tx = zx0 + (i as f64 + 0.5) * cell;
            let ty = zy0 + (N as f64 - 1.0 - j as f64 + 0.5) * cell - 15.0;
            ctx.text(&format!("{:+.1}", d), tx, ty);
        }
    }

    // zero contour through the patch: marching squares over the
    // sample grid, linear interpolation on each crossed edge.
    // Sample (i, j) sits at the center of its cell.
    let pos = |i: f64, j: f64| {
        (
            zx0 + (i + 0.5) * cell,
            zy0 + (N as f64 - 1.0 - j + 0.5) * cell,
        )
    };
    ctx.no_fill().stroke(Color::rgb(239, 68, 68)).stroke_width(8.0);
    for j in 0..N - 1 {
        for i in 0..N - 1 {
            let corners = [
                (i, j, at(px0 + i, py0 + j)),
                (i + 1, j, at(px0 + i + 1, py0 + j)),
                (i + 1, j + 1, at(px0 + i + 1, py0 + j + 1)),
                (i, j + 1, at(px0 + i, py0 + j + 1)),
            ];
            let mut pts: Vec<(f64, f64)> = Vec::new();
            for e in 0..4 {
                let (ia, ja, va) = corners[e];
                let (ib, jb, vb) = corners[(e + 1) % 4];
                let (va, vb) = (va as f64 - 128.0, vb as f64 - 128.0);
                if (va >= 0.0) != (vb >= 0.0) {
                    let t = va / (va - vb);
                    let fi = ia as f64 + (ib as f64 - ia as f64) * t;
                    let fj = ja as f64 + (jb as f64 - ja as f64) * t;
                    pts.push(pos(fi, fj));
                }
            }
            if pts.len() == 2 {
                ctx.line(pts[0].0, pts[0].1, pts[1].0, pts[1].1);
            } else if pts.len() == 4 {
                ctx.line(pts[0].0, pts[0].1, pts[1].0, pts[1].1);
                ctx.line(pts[2].0, pts[2].1, pts[3].0, pts[3].1);
            }
        }
    }

    let mut renderer = Renderer::new(W as u32, H as u32);
    renderer
        .load_font(concat!(
            env!("HOME"),
            "/GH/repos/designbot/designbot-render/assets/IBMPlexMono-Regular.ttf"
        ))
        .expect("mono font");
    renderer
        .render_to_png(&ctx, "ntf-sdf-how.png")
        .expect("render failed");
    println!("Rendered ntf-sdf-how.png patch at ({}, {})", px0, py0);
}
