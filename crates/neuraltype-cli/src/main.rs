//! NeuralType CLI: train the neural font from the procedural teacher,
//! and render proof sheets / text samples to SVG.
//!
//!   ntf train [out.ntf]           — distill teacher → .ntf font
//!   ntf sheet [font.ntf] [out.svg]— proof sheet, teacher vs model
//!   ntf render <text> [font.ntf] [out.svg] [elong]

use neuraltype_core::art::{self, ALPHABET};
use neuraltype_core::model::{self, Mlp, NeuralFont};
use neuraltype_core::shape::{layout, Dir, GlyphSource, Line};
use neuraltype_core::{font, Form, GlyphImage, MAX_ELONG};

/// The teacher as a glyph source.
struct Teacher;
impl GlyphSource for Teacher {
    fn glyph(&self, ch: char, form: Form, elongation: f64) -> Option<GlyphImage> {
        let (class, dots) = art::letter_of_char(ch)?;
        art::render(class, dots, form, elongation.round() as usize)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("train") => train(args.get(1).map_or("build/kufic.ntf", |s| s)),
        Some("sheet") => sheet(
            args.get(1).map_or("build/kufic.ntf", |s| s),
            args.get(2).map_or("build/sheet.svg", |s| s),
        ),
        Some("render") => {
            let text = args.get(1).expect("usage: ntf render <text> [font] [out] [elong]");
            render(
                text,
                args.get(2).map_or("build/kufic.ntf", |s| s),
                args.get(3).map_or("build/render.svg", |s| s),
                args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0),
            )
        }
        _ => eprintln!("usage: ntf train|sheet|render ..."),
    }
}

/// Build the full training set: every supported (letter, form,
/// elongation) context the teacher can draw.
fn dataset() -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut data = Vec::new();
    for (li, &c) in ALPHABET.iter().enumerate() {
        let (class, dots) = art::letter_of_char(c).unwrap();
        for form in Form::ALL {
            let elongs: &[usize] =
                if art::elongatable(class, form) { &[0, 1, 2, 3, 4] } else { &[0] };
            for &e in elongs {
                if let Some(img) = art::render(class, dots, form, e) {
                    data.push((
                        model::encode_input(li, form, e as f64),
                        model::encode_target(&img),
                    ));
                }
            }
        }
    }
    data
}

fn train(out_path: &str) {
    let mut data = dataset();
    // Oversample yeh: its isolated/final below-dots occupy grid rows no
    // other letter touches, and plain MSE underweights them.
    let yeh_idx = ALPHABET.iter().position(|&c| c == 'ي').unwrap();
    let extra: Vec<_> = data
        .iter()
        .filter(|(x, _)| x[yeh_idx] == 1.0)
        .cloned()
        .collect();
    for _ in 0..3 {
        data.extend(extra.iter().cloned());
    }
    println!("dataset: {} samples", data.len());
    let mut mlp = Mlp::new_font(42);
    println!("model: {} params (~{} KB as f32)", mlp.n_params(), mlp.n_params() * 4 / 1024);

    // Adam, full-batch.
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut m: Vec<(Vec<f32>, Vec<f32>)> = mlp
        .layers
        .iter()
        .map(|l| (vec![0.0; l.w.len()], vec![0.0; l.b.len()]))
        .collect();
    let mut v = m.clone();
    let epochs = 26000;
    for epoch in 1..=epochs {
        let mut grads: Vec<(Vec<f32>, Vec<f32>)> = mlp
            .layers
            .iter()
            .map(|l| (vec![0.0; l.w.len()], vec![0.0; l.b.len()]))
            .collect();
        let mut loss = 0.0;
        for (x, t) in &data {
            loss += mlp.backward(x, t, &mut grads);
        }
        loss /= data.len() as f32;
        let scale = 1.0 / data.len() as f32;
        let t = epoch as i32;
        let (c1, c2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
        // Step-decayed learning rate: coarse fit, then settle exactly.
        let lr: f32 = if epoch <= 6000 { 3e-3 } else if epoch <= 12000 { 1e-3 } else if epoch <= 20000 { 3e-4 } else { 1e-4 };
        for (li, l) in mlp.layers.iter_mut().enumerate() {
            let (gw, gb) = &grads[li];
            let (mw, mb) = &mut m[li];
            let (vw, vb) = &mut v[li];
            for i in 0..l.w.len() {
                let g = gw[i] * scale;
                mw[i] = b1 * mw[i] + (1.0 - b1) * g;
                vw[i] = b2 * vw[i] + (1.0 - b2) * g * g;
                l.w[i] -= lr * (mw[i] / c1) / ((vw[i] / c2).sqrt() + eps);
            }
            for i in 0..l.b.len() {
                let g = gb[i] * scale;
                mb[i] = b1 * mb[i] + (1.0 - b1) * g;
                vb[i] = b2 * vb[i] + (1.0 - b2) * g * g;
                l.b[i] -= lr * (mb[i] / c1) / ((vb[i] / c2).sqrt() + eps);
            }
        }
        if epoch % 500 == 0 || epoch == 1 {
            println!("epoch {epoch:5}  loss {loss:.6}");
        }
    }

    let nf = NeuralFont { mlp, alphabet: ALPHABET.to_vec() };
    // Accuracy: fraction of contexts reproduced cell-exactly.
    let (mut exact, mut total, mut cells_wrong) = (0usize, 0usize, 0usize);
    for (li, &c) in ALPHABET.iter().enumerate() {
        let _ = li;
        let (class, dots) = art::letter_of_char(c).unwrap();
        for form in Form::ALL {
            let elongs: &[usize] =
                if art::elongatable(class, form) { &[0, 1, 2, 3, 4] } else { &[0] };
            for &e in elongs {
                let Some(want) = art::render(class, dots, form, e) else { continue };
                let got = nf.glyph(c, form, e as f64).unwrap();
                total += 1;
                let wrong = want
                    .cells
                    .iter()
                    .zip(&got.cells)
                    .filter(|(a, b)| a != b)
                    .count();
                cells_wrong += wrong;
                if wrong == 0 && want.advance == got.advance {
                    exact += 1;
                } else {
                    println!("  miss: {c} {form:?} e={e} cells={wrong} adv want {} got {}", want.advance, got.advance);
                }
            }
        }
    }
    println!("fidelity: {exact}/{total} contexts cell-exact, {cells_wrong} cells wrong overall");

    let bytes = font::save(&nf, "square-kufic");
    std::fs::create_dir_all(std::path::Path::new(out_path).parent().unwrap()).unwrap();
    std::fs::write(out_path, &bytes).unwrap();
    println!("wrote {out_path} ({} bytes)", bytes.len());
}

fn svg_of_lines(lines: &[(String, Line)], scale: f64) -> String {
    let mut maxw: f64 = 0.0;
    for (_, l) in lines {
        maxw = maxw.max(l.width);
    }
    let row_h = 18.0; // grid rows + margin
    let h = lines.len() as f64 * row_h * scale;
    let w = (maxw + 4.0) * scale;
    let mut s = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/>"#
    );
    for (i, (label, line)) in lines.iter().enumerate() {
        let ty = i as f64 * row_h * scale + 2.0 * scale;
        s.push_str(&format!(
            r#"<g transform="translate({x},{ty}) scale({scale})">"#,
            x = (maxw - line.width + 2.0) * scale, // right-align
        ));
        s.push_str(&format!(
            r#"<path d="{}" fill="black" fill-rule="nonzero"/>"#,
            line.path.to_svg()
        ));
        s.push_str("</g>");
        s.push_str(&format!(
            r##"<text x="4" y="{}" font-size="10" fill="#888" font-family="monospace">{}</text>"##,
            ty + 8.0 * scale,
            label
        ));
    }
    s.push_str("</svg>");
    s
}

/// Proof sheet: each letter in all forms, teacher vs model.
fn sheet(font_path: &str, out_path: &str) {
    let nf = font::load(&std::fs::read(font_path).expect("font not found — run `ntf train`"))
        .unwrap();
    let mut lines = Vec::new();
    for &c in ALPHABET {
        // A carrier context showing iso, init+med+fin: "c  ccc" style.
        let sample = format!("{c} {c}{c}{c}");
        lines.push((format!("teacher {c}"), layout(&Teacher, &sample, 0.0, Dir::Auto)));
        lines.push((format!("model   {c}"), layout(&nf, &sample, 0.0, Dir::Auto)));
    }
    std::fs::write(out_path, svg_of_lines(&lines, 8.0)).unwrap();
    println!("wrote {out_path}");
}

fn render(text: &str, font_path: &str, out_path: &str, elong: f64) {
    let nf = font::load(&std::fs::read(font_path).expect("font not found — run `ntf train`"))
        .unwrap();
    let elong = elong.clamp(0.0, MAX_ELONG as f64);
    let lines = vec![
        ("model".to_string(), layout(&nf, text, elong, Dir::Auto)),
        ("teacher".to_string(), layout(&Teacher, text, elong, Dir::Auto)),
    ];
    std::fs::write(out_path, svg_of_lines(&lines, 10.0)).unwrap();
    println!("wrote {out_path}");
}
