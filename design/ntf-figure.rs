// Generic blog-figure renderer: blits an RGBA blob from figures.py
// onto the house dark background. The blob is upscaled by an integer
// factor with nearest-neighbor sampling and drawn 1:1, so the
// low-resolution pixel grid stays crisp — no bilinear blur.
// args: <blob.rgba> <w> <h> [margin_px] [max_width]
use designbot::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (path, w, h): (&str, usize, usize) =
        (&args[1], args[2].parse().unwrap(), args[3].parse().unwrap());
    let margin: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(80);
    let max_w: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(2400);
    let data = std::fs::read(path).expect("rgba blob");
    assert_eq!(data.len(), w * h * 4);

    let s = ((max_w - 2 * margin) / w).max(1);
    let (uw, uh) = (w * s, h * s);
    let mut up = vec![0u8; uw * uh * 4];
    for y in 0..uh {
        for x in 0..uw {
            let src = ((y / s) * w + x / s) * 4;
            let dst = (y * uw + x) * 4;
            up[dst..dst + 4].copy_from_slice(&data[src..src + 4]);
        }
    }

    let cw = (uw + 2 * margin) as f64;
    let ch = (uh + 2 * margin) as f64;
    let mut ctx = Canvas::new(cw, ch);
    ctx.background(Color::rgb(12, 12, 12));
    ctx.image_rgba(up, uw as u32, uh as u32, margin as f64, margin as f64, 1.0);

    let renderer = Renderer::new(cw as u32, ch as u32);
    renderer.render_to_png(&ctx, "figure.png").expect("render failed");
    println!("Rendered figure.png {}x{} (scale {})", cw, ch, s);
}
