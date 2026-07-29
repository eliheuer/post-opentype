// OG image for the NeuralType blog post.
//
// The word قلم (qalam, "pen") set in the generative square-Kufic font,
// on its design grid, shown as live text being edited: the last letter
// (م, leftmost — RTL) carries a selection highlight and the cursor sits
// at its trailing edge. Bottom-right: a miniature neural network whose
// neurons are grid squares — the font is the network.
//
// Cell data is the engine's actual composed output for "قلم"
// (13×14 cells, from `ntf dump`).
use designbot::prelude::*;

// Composed line bitmap, row 0 = top (flipped to y-up when drawing).
const WORD: [&str; 14] = [
    ".............",
    ".............",
    ".......#.....",
    ".......#.....",
    ".......#.....",
    ".......#.....",
    ".......#.#.#.",
    ".......#.....",
    ".###...#..###",
    ".#.#...#..#.#",
    "#############",
    "#............",
    "#............",
    ".............",
];
const COLS: usize = 13;
const ROWS: usize = 14;

fn main() {
    const W: f64 = 2400.0;
    const H: f64 = 1260.0;
    let mut ctx = Canvas::new(W, H);
    ctx.background(Color::rgb(12, 12, 12));

    let gold = Color::rgb(232, 184, 75);
    let grid_line = Color::rgba(255, 255, 255, 22);
    let grid_line_major = Color::rgba(255, 255, 255, 44);
    let wire = Color::rgb(95, 95, 95);

    // Word placement: fit 14 rows comfortably.
    let cell: f64 = 76.0;
    let wx = (W - COLS as f64 * cell) / 2.0 - 60.0; // slight nudge; NN sits right
    let wy = (H - ROWS as f64 * cell) / 2.0;

    // Full-canvas grid aligned to the word's cells (graph paper).
    ctx.stroke(grid_line).stroke_width(1.5).no_fill();
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
    // Slightly brighter grid over the word's own canvas.
    ctx.stroke(grid_line_major).stroke_width(1.5).no_fill();
    for i in 0..=COLS {
        let x = wx + i as f64 * cell;
        ctx.line(x, wy, x, wy + ROWS as f64 * cell);
    }
    for j in 0..=ROWS {
        let y = wy + j as f64 * cell;
        ctx.line(wx, y, wx + COLS as f64 * cell, y);
    }

    // Selection highlight over the last letter م (RTL: leftmost,
    // columns 0..5 of the word), full line height — editor style.
    ctx.no_stroke();
    ctx.fill(Color::rgba(232, 184, 75, 38));
    ctx.rect(wx, wy, 5.0 * cell, ROWS as f64 * cell);

    // The word: engine cells, filled gold. Row 0 is the top row, and
    // designbot's y axis points up, so flip.
    ctx.fill(gold).no_stroke();
    for (ry, row) in WORD.iter().enumerate() {
        for (cx, ch) in row.bytes().enumerate() {
            if ch == b'#' {
                let x = wx + cx as f64 * cell;
                let y = wy + (ROWS - 1 - ry) as f64 * cell;
                ctx.rect(x, y, cell, cell);
            }
        }
    }

    // Cursor: after the last letter of RTL text = at its left edge.
    ctx.fill(gold).no_stroke();
    ctx.rect(wx - 7.0, wy, 14.0, ROWS as f64 * cell);

    // Mini neural network, bottom-right: neurons are grid squares.
    let layers: [&[f64]; 3] = [&[0.0, 1.0, 2.0], &[-0.5, 0.5, 1.5, 2.5], &[0.5, 1.5]];
    let nn_x = W - 420.0;
    let nn_y = 120.0;
    let dx = 130.0;
    let dy = 96.0;
    let node = 34.0;
    let center = |l: usize, t: f64| -> (f64, f64) { (nn_x + l as f64 * dx, nn_y + t * dy) };
    // Wires first, behind the nodes.
    ctx.stroke(wire).stroke_width(2.5).no_fill();
    for l in 0..layers.len() - 1 {
        for &a in layers[l] {
            for &b in layers[l + 1] {
                let (x1, y1) = center(l, a);
                let (x2, y2) = center(l + 1, b);
                ctx.line(x1, y1, x2, y2);
            }
        }
    }
    // Square neurons.
    ctx.fill(gold).no_stroke();
    for (l, ts) in layers.iter().enumerate() {
        for &t in ts.iter() {
            let (cx, cy) = center(l, t);
            ctx.rect(cx - node / 2.0, cy - node / 2.0, node, node);
        }
    }

    let renderer = Renderer::new(W as u32, H as u32);
    renderer
        .render_to_png(&ctx, "ntf-og.png")
        .expect("render failed");
    println!("Rendered ntf-og.png");
}
