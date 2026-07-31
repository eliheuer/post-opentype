// Illustration for the "Fields" section of the Nastaliq Distilled
// blog post: the isolated خ from Gulzar (left) and the same shape as
// the signed-distance field the model actually learns (right).
//
// Reads the real training data: shape 6 of fields.bin (u8 SDF,
// 155x219, 128 = on the contour, spread = 1/8 em).
use designbot::prelude::*;

const W_PX: usize = 155;
const H_PX: usize = 219;
const SHAPE: usize = 6; // isolated خ
// crop window around the shape (bbox x 58..103, y 48..130, spread 8px)
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

    const W: f64 = 2400.0;
    const H: f64 = 1260.0;
    let mut ctx = Canvas::new(W, H);
    ctx.background(Color::rgb(12, 12, 12));

    let scale = 8.0; // integer: panels are blitted pre-upscaled, no resampling blur
    let pw = CW as f64 * scale;
    let ph = CH as f64 * scale;
    let gap = (W - 2.0 * pw) / 3.0;
    let y0 = (H - ph) / 2.0;

    // Left: the figure. Inside pixels in the demo's ink green.
    let mut left = vec![0u8; CW * CH * 4];
    // Right: the field itself, the grayscale grid, with the zero
    // contour (v = 128) marked in the demo's red.
    let mut right = vec![0u8; CW * CH * 4];
    for cy in 0..CH {
        for cx in 0..CW {
            let (x, y) = (CX + cx, CY + cy);
            let i = cy * CW + cx;
            let v = f[y * W_PX + x];
            if v >= 128 {
                left[i * 4..i * 4 + 4].copy_from_slice(&[42, 163, 95, 255]);
            }
            // zero contour: inside pixel with an outside 4-neighbor
            let mut edge = false;
            if v >= 128 {
                for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= W_PX as i64 || ny >= H_PX as i64 {
                        continue;
                    }
                    if f[ny as usize * W_PX + nx as usize] < 128 {
                        edge = true;
                    }
                }
            }
            let px = if edge {
                [239, 68, 68, 255]
            } else {
                // quantize the SDF into 8 hard gray bands: the smooth
                // ramp reads as blur, the bands read as a pixel grid
                let q = (v / 32) as u32;
                let g = (25 + q * 27) as u8;
                [g, g, g, 255]
            };
            right[i * 4..i * 4 + 4].copy_from_slice(&px);
        }
    }

    // panel frames
    ctx.no_fill().stroke(Color::rgb(70, 70, 70)).stroke_width(2.0);
    ctx.rect(gap, y0, pw, ph);
    ctx.rect(gap * 2.0 + pw, y0, pw, ph);

    // nearest-neighbor upscale, drawn 1:1 so pixels stay crisp
    let s = scale as usize;
    let up = |src: &[u8]| {
        let (uw, uh) = (CW * s, CH * s);
        let mut out = vec![0u8; uw * uh * 4];
        for y in 0..uh {
            for x in 0..uw {
                let i = ((y / s) * CW + x / s) * 4;
                let o = (y * uw + x) * 4;
                out[o..o + 4].copy_from_slice(&src[i..i + 4]);
            }
        }
        out
    };
    ctx.image_rgba(up(&left), (CW * s) as u32, (CH * s) as u32, gap, y0, 1.0);
    ctx.image_rgba(up(&right), (CW * s) as u32, (CH * s) as u32, gap * 2.0 + pw, y0, 1.0);

    let renderer = Renderer::new(W as u32, H as u32);
    renderer
        .render_to_png(&ctx, "ntf-sdf.png")
        .expect("render failed");
    println!("Rendered ntf-sdf.png");
}
