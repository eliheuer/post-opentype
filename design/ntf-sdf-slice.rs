// "A signed-distance field in one dimension" figure for the Shapes
// as Fields section of the Nasta’liq Distilled blog post.
//
// Left: the isolated خ as its distance field, with a red horizontal
// line marking one row of cells. Right: the values in that row
// plotted as a curve. The curve rises inside the strokes and falls
// outside; the letter's edge is exactly where it crosses zero.
//
// Reads shape 6 of fields.bin (u8 SDF, 155x219, 128 = on the
// contour, spread 1/8 em = 8 px at 64 px/em).
use designbot::prelude::*;

const W_PX: usize = 155;
const H_PX: usize = 219;
const SHAPE: usize = 6; // isolated خ
// same crop window as ntf-sdf.rs (bbox x 58..103, y 48..130)
const CX: usize = 38;
const CY: usize = 28;
const CW: usize = 86;
const CH: usize = 122;

fn main() {
    let bin = std::fs::read(concat!(
        env!("HOME"),
        "/GH/repos/post-opentype/data/fields-gulzar-64/fields.bin"
    ))
    .expect("fields.bin");
    let f = &bin[SHAPE * W_PX * H_PX..(SHAPE + 1) * W_PX * H_PX];
    let at = |x: usize, y: usize| f[y * W_PX + x];
    let d_of = |v: u8| (v as f64 - 128.0) / 16.0; // signed px

    // Pick the slice row: crosses the bowl twice (four zero
    // crossings) with the deepest inside values.
    let mut row = CY + CH / 2;
    let mut best = f64::MIN;
    for y in CY..CY + CH {
        let mut xs: Vec<usize> = Vec::new();
        let mut depth = 0.0f64;
        for x in CX..CX + CW - 1 {
            let (a, b) = (d_of(at(x, y)), d_of(at(x + 1, y)));
            if (a >= 0.0) != (b >= 0.0) {
                xs.push(x);
            }
            if a > 0.0 {
                depth += a;
            }
        }
        if xs.len() != 4 {
            continue;
        }
        // the dip between the two strokes must clearly go below
        // zero, so the plot reads: inside, outside, inside
        let mut dip = 0.0f64;
        for x in xs[1]..=xs[2] {
            dip = dip.min(d_of(at(x, y)));
        }
        if dip <= -1.5 && depth > best {
            best = depth;
            row = y;
        }
    }

    const W: f64 = 2400.0;
    const H: f64 = 1260.0;
    let mut ctx = Canvas::new(W, H);
    ctx.background(Color::rgb(12, 12, 12));

    // ---- left panel: the field, cell by cell, y-flipped ----
    let scale = 8.0;
    let pw = CW as f64 * scale;
    let ph = CH as f64 * scale;
    let gap = 120.0;
    let lx0 = gap;
    let ly0 = (H - ph) / 2.0;
    for cy in 0..CH {
        for cx in 0..CW {
            let v = at(CX + cx, CY + cy);
            let q = (v / 32) as u32;
            let g = (25 + q * 27) as u8;
            ctx.no_stroke().fill(Color::rgb(g, g, g));
            ctx.rect(
                lx0 + cx as f64 * scale,
                ly0 + (CH - 1 - cy) as f64 * scale,
                scale,
                scale,
            );
        }
    }
    ctx.no_fill().stroke(Color::rgb(70, 70, 70)).stroke_width(2.0);
    ctx.rect(lx0, ly0, pw, ph);
    // the slice row, in red
    let ry = ly0 + (CH as f64 - (row - CY) as f64 - 0.5) * scale;
    ctx.no_fill().stroke(Color::rgb(239, 68, 68)).stroke_width(5.0);
    ctx.line(lx0 - 18.0, ry, lx0 + pw + 18.0, ry);

    // ---- right panel: the row's values as a curve ----
    let px0 = W - gap - pw; // same width as the left panel
    let py0 = ly0;
    let pph = ph;
    // value range: fields span about -8..+8 px; show -8..+8
    let vmax = 8.0f64;
    let y_of = |d: f64| py0 + pph / 2.0 + (d / vmax) * (pph / 2.0 - 30.0);
    let x_of = |cx: f64| px0 + (cx / CW as f64) * pw;
    ctx.no_fill().stroke(Color::rgb(70, 70, 70)).stroke_width(2.0);
    ctx.rect(px0, py0, pw, pph);
    // zero axis: the letter's edge lives here
    ctx.no_fill().stroke(Color::rgb(160, 160, 160)).stroke_width(2.5);
    ctx.line(px0, y_of(0.0), px0 + pw, y_of(0.0));
    // the value curve, green like the ink
    ctx.no_fill().stroke(Color::rgb(42, 163, 95)).stroke_width(6.0);
    for cx in 0..CW - 1 {
        let a = d_of(at(CX + cx, row));
        let b = d_of(at(CX + cx + 1, row));
        ctx.line(
            x_of(cx as f64 + 0.5),
            y_of(a),
            x_of(cx as f64 + 1.5),
            y_of(b),
        );
    }
    // zero crossings: red dots on the axis, the edges of the letter
    ctx.no_stroke().fill(Color::rgb(239, 68, 68));
    for cx in 0..CW - 1 {
        let a = d_of(at(CX + cx, row));
        let b = d_of(at(CX + cx + 1, row));
        if (a >= 0.0) != (b >= 0.0) {
            let t = a / (a - b);
            ctx.oval(x_of(cx as f64 + 0.5 + t) - 9.0, y_of(0.0) - 9.0, 18.0, 18.0);
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
        .render_to_png(&ctx, "ntf-sdf-slice.png")
        .expect("render failed");
    println!("Rendered ntf-sdf-slice.png row {}", row);
}
