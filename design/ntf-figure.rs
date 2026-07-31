// Generic blog-figure renderer: blits an RGBA blob from figures.py
// onto the house dark background, scaled to a 2400px-wide canvas.
// args: <blob.rgba> <w> <h> [margin_px]
use designbot::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (path, w, h): (&str, usize, usize) =
        (&args[1], args[2].parse().unwrap(), args[3].parse().unwrap());
    let margin: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(80.0);
    let data = std::fs::read(path).expect("rgba blob");
    assert_eq!(data.len(), w * h * 4);

    const CW: f64 = 2400.0;
    let scale = (CW - 2.0 * margin) / w as f64;
    let ch = (h as f64 * scale + 2.0 * margin).ceil();
    let mut ctx = Canvas::new(CW, ch);
    ctx.background(Color::rgb(12, 12, 12));
    ctx.translate(margin, margin);
    ctx.scale(scale);
    ctx.image_rgba(data, w as u32, h as u32, 0.0, 0.0, 1.0);

    let renderer = Renderer::new(CW as u32, ch as u32);
    renderer.render_to_png(&ctx, "figure.png").expect("render failed");
    println!("Rendered figure.png {}x{}", CW, ch);
}
