// Dump the composed cell grid + spans for a word (design-asset helper).
use neuraltype_core::shape::{layout, Dir, GlyphSource};
struct T;
impl GlyphSource for T {
    fn glyph(&self, c: char, form: neuraltype_core::Form, e: f64) -> Option<neuraltype_core::GlyphImage> {
        let (cl, d) = neuraltype_core::art::letter_of_char(c)?;
        neuraltype_core::art::render(cl, d, form, e as usize)
    }
}
fn main() {
    let text = std::env::args().nth(1).unwrap_or("قلم".into());
    let line = layout(&T, &text, 0.0, Dir::Auto);
    let w = line.width.round() as usize;
    println!("w={w} h={}", neuraltype_core::GRID_H);
    for s in &line.spans { println!("span i={} x={} w={}", s.index, s.x, s.width); }
    // outline loops as corner-point lists (for design assets)
    let mut loop_pts: Vec<(f64, f64)> = Vec::new();
    for el in line.path.elements() {
        match el {
            kurbo::PathEl::MoveTo(p) => {
                loop_pts = vec![(p.x, p.y)];
            }
            kurbo::PathEl::LineTo(p) => loop_pts.push((p.x, p.y)),
            kurbo::PathEl::ClosePath => {
                let s: Vec<String> =
                    loop_pts.iter().map(|(x, y)| format!("({x:.0},{y:.0})")).collect();
                println!("loop {}", s.join(" "));
            }
            _ => {}
        }
    }
    use kurbo::Shape;
    for y in 0..neuraltype_core::GRID_H {
        let row: String = (0..w).map(|x| {
            let pt = kurbo::Point::new(x as f64 + 0.5, y as f64 + 0.5);
            if line.path.winding(pt) != 0 { '#' } else { '.' }
        }).collect();
        println!("{row}");
    }
}
