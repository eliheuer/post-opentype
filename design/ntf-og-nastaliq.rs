// OG image for the "Nastaliq Distilled" post: the cascade figure
// (distill cascade output for نستعليق) centered on a 2400x1260
// social-card canvas, nearest-neighbor upscaled so the pixel grid
// stays crisp. args: <blob.rgba> <w> <h>
use designbot::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (path, w, h): (&str, usize, usize) =
        (&args[1], args[2].parse().unwrap(), args[3].parse().unwrap());
    let data = std::fs::read(path).expect("rgba blob");
    assert_eq!(data.len(), w * h * 4);

    const CW: f64 = 2400.0;
    const CH: f64 = 1260.0;
    let s = ((CH as usize - 90) / h).min((CW as usize - 90) / w).max(1);
    let (uw, uh) = (w * s, h * s);
    let mut up = vec![0u8; uw * uh * 4];
    for y in 0..uh {
        for x in 0..uw {
            let src = ((y / s) * w + x / s) * 4;
            let dst = (y * uw + x) * 4;
            up[dst..dst + 4].copy_from_slice(&data[src..src + 4]);
        }
    }

    // find the baseline rule row in the blob (the [70,70,70] line)
    let rule_row = (0..h)
        .max_by_key(|&y| {
            (0..w)
                .filter(|&x| {
                    let i = (y * w + x) * 4;
                    data[i..i + 3] == [70, 70, 70]
                })
                .count()
        })
        .unwrap();

    let mut ctx = Canvas::new(CW, CH);
    ctx.background(Color::rgb(12, 12, 12));
    // full-width baseline, aligned with the blob's rule
    let y0 = ((CH - uh as f64) / 2.0).round();
    let rule_y = y0 + uh as f64 - (rule_row as f64 + 0.5) * s as f64;
    ctx.stroke(Color::rgb(70, 70, 70)).stroke_width(s as f64).no_fill();
    ctx.line(0.0, rule_y, CW, rule_y);
    ctx.image_rgba(
        up,
        uw as u32,
        uh as u32,
        ((CW - uw as f64) / 2.0).round(),
        y0,
        1.0,
    );

    let renderer = Renderer::new(CW as u32, CH as u32);
    renderer.render_to_png(&ctx, "ntf-og-nastaliq.png").expect("render failed");
    println!("Rendered ntf-og-nastaliq.png (scale {s})");
}
