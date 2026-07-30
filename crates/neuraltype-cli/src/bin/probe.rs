// Stage-0 spike: measure Amiri's contextual richness with rustybuzz.
//
// Shapes an all-pairs corpus and counts, for each Arabic letter and
// joining position, how many distinct glyphs Amiri actually uses
// depending on the neighbor. That count is the true denominator for
// the naskh-distilled model (docs/DISTILL.md step 1).
use std::collections::{BTreeMap, HashSet};

const LETTERS: &[char] = &[
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ',
    'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'ة', 'و', 'ي', 'ى', 'ء',
];

fn main() {
    let bytes = std::fs::read("data/Amiri-Regular.ttf").expect("run from repo root");
    let face = rustybuzz::Face::from_slice(&bytes, 0).expect("bad font");

    let shape = |text: &str| -> Vec<(u32, u32)> {
        let mut buf = rustybuzz::UnicodeBuffer::new();
        buf.push_str(text);
        let out = rustybuzz::shape(&face, &[], buf);
        out.glyph_infos()
            .iter()
            .map(|g| (g.glyph_id, g.cluster))
            .collect()
    };

    // For every ordered pair (a, b): shape "ab". Record which glyph a
    // takes before b, and which glyph b takes after a.
    let mut first_glyphs: BTreeMap<char, HashSet<u32>> = BTreeMap::new();
    let mut second_glyphs: BTreeMap<char, HashSet<u32>> = BTreeMap::new();
    let mut ligature_pairs = 0usize;
    let mut total_pairs = 0usize;
    for &a in LETTERS {
        for &b in LETTERS {
            total_pairs += 1;
            let s: String = [a, b].iter().collect();
            let glyphs = shape(&s);
            let split = a.len_utf8() as u32;
            let a_glyphs: Vec<u32> =
                glyphs.iter().filter(|(_, c)| *c < split).map(|(g, _)| *g).collect();
            let b_glyphs: Vec<u32> =
                glyphs.iter().filter(|(_, c)| *c >= split).map(|(g, _)| *g).collect();
            if a_glyphs.is_empty() || b_glyphs.is_empty() {
                ligature_pairs += 1; // fused into one cluster
                continue;
            }
            first_glyphs.entry(a).or_default().extend(a_glyphs);
            second_glyphs.entry(b).or_default().extend(b_glyphs);
        }
    }

    println!("Amiri-Regular: glyph count {}", face.number_of_glyphs());
    println!("pairs shaped: {total_pairs}, fused into ligatures: {ligature_pairs}");
    println!();
    println!("letter  before-variants  after-variants");
    let mut total_first = 0usize;
    let mut total_second = 0usize;
    for &c in LETTERS {
        let f = first_glyphs.get(&c).map_or(0, |s| s.len());
        let g = second_glyphs.get(&c).map_or(0, |s| s.len());
        total_first += f;
        total_second += g;
        println!("{c}       {f:<16} {g}");
    }
    println!();
    println!("distinct first-position glyph variants: {total_first}");
    println!("distinct second-position glyph variants: {total_second}");
}
