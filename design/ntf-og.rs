// OG image for the NeuralType blog post: the word قلم (qalam, "pen")
// exactly as the demo's structure view draws it. Green letterform on
// the design grid, light-gray traced outlines with red corner points,
// and the red text cursor at the trailing edge of the last letter.
//
// Cell and outline data are the engine's actual output for "قلم"
// (13×14 cells; loops from `ntf dump`). Grid coordinates are y-down;
// designbot is y-up, so vertices flip through ROWS.
use designbot::prelude::*;

const WORD: [&str; 14] = [
    ".............",
    ".............",
    ".......#.....",
    ".......#.....",
    ".......#.....",
    ".......#.....",
    ".......#..#.#",
    ".......#.....",
    ".###...#..###",
    ".#.#...#..#.#",
    "#############",
    "#............",
    "#............",
    ".............",
];
const LOOPS: [&[(f64, f64)]; 5] = [
    &[(4.,8.),(4.,10.),(7.,10.),(7.,2.),(8.,2.),(8.,10.),(10.,10.),(10.,8.),(13.,8.),(13.,11.),(1.,11.),(1.,13.),(0.,13.),(0.,10.),(1.,10.),(1.,8.)],
    &[(11.,10.),(12.,10.),(12.,9.),(11.,9.)],
    &[(11.,7.),(10.,7.),(10.,6.),(11.,6.)],
    &[(12.,7.),(12.,6.),(13.,6.),(13.,7.)],
    &[(2.,10.),(3.,10.),(3.,9.),(2.,9.)],
];
const COLS: usize = 13;
const ROWS: usize = 14;

fn main() {
    const W: f64 = 2400.0;
    const H: f64 = 1260.0;
    let mut ctx = Canvas::new(W, H);
    ctx.background(Color::rgb(12, 12, 12));

    let green = Color::rgb(42, 163, 95);
    let red = Color::rgb(239, 68, 68);
    let outline = Color::rgb(229, 229, 229);
    let grid = Color::rgb(70, 70, 70);

    let cell: f64 = 80.0;
    let wx = (W - COLS as f64 * cell) / 2.0;
    let wy = (H - ROWS as f64 * cell) / 2.0;
    // grid vertex (y-down) -> canvas point (y-up)
    let vx = |x: f64| wx + x * cell;
    let vy = |y: f64| wy + (ROWS as f64 - y) * cell;

    // Full-canvas grid aligned to the word's cells.
    ctx.stroke(grid).stroke_width(2.0).no_fill();
    let mut x = wx % cell;
    while x <= W {
        ctx.line(x, 0.0, x, H);
        x += cell;
    }
    let mut y = wy % cell;
    while y <= H {
        ctx.line(0.0, y, W, y);
        y += cell;
    }

    // Letterform cells, filled green.
    ctx.fill(green).no_stroke();
    for (ry, row) in WORD.iter().enumerate() {
        for (cx, ch) in row.bytes().enumerate() {
            if ch == b'#' {
                ctx.rect(vx(cx as f64), vy(ry as f64 + 1.0), cell, cell);
            }
        }
    }

    // Traced outlines, stroked light gray.
    ctx.stroke(outline).stroke_width(6.0).no_fill();
    for pts in LOOPS {
        let poly: Vec<(f64, f64)> = pts.iter().map(|&(x, y)| (vx(x), vy(y))).collect();
        ctx.polygon(&poly, true);
    }

    // Corner points, red squares.
    ctx.fill(red).no_stroke();
    let ps = 22.0;
    for pts in LOOPS {
        for &(x, y) in pts.iter() {
            ctx.rect(vx(x) - ps / 2.0, vy(y) - ps / 2.0, ps, ps);
        }
    }

    // Text cursor at the trailing (left) edge of the last letter:
    // red I-beam with inward-facing triangles, as in the demo.
    let cx0 = vx(0.0);
    let top = vy(0.0);
    let bot = vy(ROWS as f64);
    ctx.fill(red).no_stroke();
    ctx.rect(cx0 - 5.0, bot, 10.0, top - bot);
    ctx.polygon(&[(cx0 - 22.0, top), (cx0 + 22.0, top), (cx0, top - 30.0)], true);
    ctx.polygon(&[(cx0 - 22.0, bot), (cx0 + 22.0, bot), (cx0, bot + 30.0)], true);

    let renderer = Renderer::new(W as u32, H as u32);
    renderer
        .render_to_png(&ctx, "ntf-og.png")
        .expect("render failed");
    println!("Rendered ntf-og.png");
}
