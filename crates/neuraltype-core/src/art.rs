//! The "teacher": procedural square-Kufic letterforms on a grid.
//!
//! Each letter class is authored as ASCII art (one string per grid
//! row, `#` = filled cell), exactly like a classic square-Kufic chart.
//! The teacher composes: base form art + i'jam dots + kashida
//! elongation → a `GlyphImage`. The neural font is trained purely on
//! this output; at runtime only the model ships.
//!
//! Conventions:
//! - Strokes are 1 cell wide; counters/gaps are 1 cell.
//! - The baseline stroke sits on `BASELINE_ROW`; rows below are the
//!   descender zone.
//! - Entry (connection from the previous letter) is at the RIGHT edge
//!   on the baseline row; exit (to the next letter) at the LEFT edge.
//! - Bodies are right-aligned on the canvas; kashida elongation
//!   inserts baseline columns at the right (entry) side.

use crate::{Form, GlyphImage, BASELINE_ROW, GRID_H, GRID_W};

/// A letter-class skeleton (rasm), shared by all letters that differ
/// only in dots (e.g. ب ت ث ن ي share teeth in initial/medial forms).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    Alef,
    Beh,
    Jeem,
    Dal,
    Reh,
    Seen,
    Sad,
    Tah,
    Ain,
    Feh,
    Qaf,
    Kaf,
    Lam,
    Meem,
    Noon,
    Heh,
    Waw,
    Yeh,
    /// Standalone hamza: a small non-joining mark on the baseline.
    Hamza,
    /// The لا ligature: lam + alef fuse into one glyph (mandatory in
    /// Arabic). Shaping substitutes the pair; the model knows it as
    /// one letter id.
    LamAlef,
    /// Latin capital, 0 = A … 25 = Z. Latin letters have one form
    /// (no joining); their elongation stretches designated horizontal
    /// strokes instead of a baseline connector.
    Latin(u8),
}

/// ASCII art for one contextual form: `rows[0]` is grid row `top`.
pub struct FormArt {
    pub top: usize,
    pub rows: &'static [&'static str],
}

/// Dots: positive = n dots above, negative = n dots below, 0 = none.
pub type Dots = i8;

/// Map a Unicode codepoint to (letter class, dots). Returns None for
/// unsupported characters.
pub fn letter_of_char(c: char) -> Option<(Class, Dots)> {
    use Class::*;
    Some(match c {
        'ا' | 'أ' | 'إ' | 'آ' => (Alef, 0),
        'ب' => (Beh, -1),
        'ت' => (Beh, 2),
        'ث' => (Beh, 3),
        'ج' => (Jeem, -1),
        'ح' => (Jeem, 0),
        'خ' => (Jeem, 1),
        'د' => (Dal, 0),
        'ذ' => (Dal, 1),
        'ر' => (Reh, 0),
        'ز' => (Reh, 1),
        'س' => (Seen, 0),
        'ش' => (Seen, 3),
        'ص' => (Sad, 0),
        'ض' => (Sad, 1),
        'ط' => (Tah, 0),
        'ظ' => (Tah, 1),
        'ع' => (Ain, 0),
        'غ' => (Ain, 1),
        'ف' => (Feh, 1),
        'ق' => (Qaf, 2),
        'ك' => (Kaf, 0),
        'ل' => (Lam, 0),
        'م' => (Meem, 0),
        'ن' => (Noon, 1),
        'ه' => (Heh, 0),
        'ة' => (Heh, 2),
        'و' | 'ؤ' => (Waw, 0),
        'ي' | 'ئ' => (Yeh, -2),
        'ى' => (Yeh, 0),
        'ء' => (Hamza, 0),
        'ﻻ' | 'ﻼ' => (LamAlef, 0), // presentation forms, typed directly
        'A'..='Z' => (Latin(c as u8 - b'A'), 0),
        'a'..='z' => (Latin(c as u8 - b'a'), 0),
        _ => return None,
    })
}

/// Canonical alphabet codepoint for a supported character: hamza
/// carriers and presentation forms fold onto the base letter the
/// model was trained on.
pub fn canonical_char(c: char) -> char {
    match c {
        'أ' | 'إ' | 'آ' => 'ا',
        'ؤ' => 'و',
        'ئ' => 'ي',
        'ﻼ' => 'ﻻ',
        'a'..='z' => c.to_ascii_uppercase(),
        _ => c,
    }
}

/// Letters that never connect to the following (left) letter.
pub fn joins_left(class: Class) -> bool {
    !matches!(
        class,
        Class::Alef
            | Class::Dal
            | Class::Reh
            | Class::Waw
            | Class::Hamza
            | Class::LamAlef
            | Class::Latin(_)
    )
}

/// Letters that can receive a connection from the preceding letter.
pub fn joins_right(class: Class) -> bool {
    !matches!(class, Class::Latin(_) | Class::Hamza)
}

/// Does elongation apply to this glyph? Arabic stretches the baseline
/// connection (kashida) on joined forms; Latin stretches designated
/// horizontal strokes of its single form.
pub fn elongatable(class: Class, form: Form) -> bool {
    match class {
        Class::Latin(i) => form == Form::Isolated && latin_art(i).1.is_some(),
        _ => form.joins_prev(),
    }
}

/// The full inventory of supported codepoints (the model's alphabet).
pub const ALPHABET: &[char] = &[
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع',
    'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'ة', 'و', 'ي', 'ى', 'ء', 'ﻻ', 'A', 'B', 'C', 'D',
    'E', 'F',
    'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z',
];

macro_rules! art {
    ($top:expr, [$($row:expr),+ $(,)?]) => {
        Some(FormArt { top: $top, rows: &[$($row),+] })
    };
}

/// The authored square-Kufic art for each (class, form).
/// None = this class does not occur in this form (right-joiners have
/// no initial/medial forms).
pub fn class_art(class: Class, form: Form) -> Option<FormArt> {
    use Class::*;
    use Form::*;
    match (class, form) {
        // Alef: tall vertical; final gains a baseline foot from the entry.
        (Alef, Isolated) => art!(2, ["#", "#", "#", "#", "#", "#", "#", "#", "#"]),
        (Alef, Final) => art!(2, ["#..", "#..", "#..", "#..", "#..", "#..", "#..", "#..", "###"]),
        (Alef, _) => None,

        // Beh body (ب ت ث teeth family): open bowl; tooth when joined.
        (Beh, Isolated) => art!(8, ["#...#", "#...#", "#####"]),
        (Beh, Final) => art!(8, ["#...#.", "#...#.", "######"]),
        (Beh, Initial) => art!(8, ["..#", "..#", "###"]),
        (Beh, Medial) => art!(8, [".#.", ".#.", "###"]),

        // Noon: like beh when joined; deep descending bowl standalone.
        (Noon, Isolated) => art!(8, ["#...#", "#...#", "#...#", "#...#", "#####"]),
        (Noon, Final) => art!(8, ["#...#.", "#...#.", "#...##", "#...#.", "#####."]),
        (Noon, Initial) => art!(8, ["..#", "..#", "###"]),
        (Noon, Medial) => art!(8, [".#.", ".#.", "###"]),

        // Yeh: tooth when joined; sweeping under-tail standalone.
        (Yeh, Isolated) => art!(10, ["..###", "..#..", "###.."]),
        (Yeh, Final) => art!(10, ["...###", "...#..", "####.."]),
        (Yeh, Initial) => art!(8, ["..#", "..#", "###"]),
        (Yeh, Medial) => art!(8, [".#.", ".#.", "###"]),

        // Jeem family (ج ح خ): top bar + open jaw; descender bowl standalone.
        (Jeem, Isolated) => art!(8, ["####.", "#....", "#####", "#....", "####."]),
        (Jeem, Final) => art!(8, ["####..", "#.....", "######", "#.....", "####.."]),
        (Jeem, Initial) => art!(8, [".####", ".#...", "#####"]),
        (Jeem, Medial) => art!(8, [".####.", ".#....", "######"]),

        // Dal: corner stroke on the baseline.
        (Dal, Isolated) => art!(8, ["...#", "...#", "####"]),
        (Dal, Final) => art!(8, ["...#.", "...#.", "#####"]),
        (Dal, _) => None,

        // Reh: corner stroke dropped below the baseline.
        (Reh, Isolated) => art!(10, ["...#", "...#", "####"]),
        (Reh, Final) => art!(10, ["...##", "...#.", "####."]),
        (Reh, _) => None,

        // Seen: three teeth; adds a descender bowl standalone.
        (Seen, Isolated) => art!(9, ["..#.#.#", "#######", "#......", "####..."]),
        (Seen, Final) => art!(9, ["..#.#.#.", "########", "#.......", "####...."]),
        (Seen, Initial) => art!(9, [".#.#.#", "######"]),
        (Seen, Medial) => art!(9, [".#.#.#", "######"]),

        // Sad: enclosed eye + baseline; descender bowl standalone.
        (Sad, Isolated) => art!(8, ["..###", "..#.#", "#####"]),
        (Sad, Final) => art!(8, ["..###.", "..#.#.", "######", "#.....", "####.."]),
        (Sad, Initial) => art!(8, ["..###", "..#.#", "#####"]),
        (Sad, Medial) => art!(8, [".###.", ".#.#.", "#####"]),

        // Tah: sad-like eye with a tall stem; body identical in all forms.
        (Tah, Isolated) | (Tah, Initial) | (Tah, Medial) | (Tah, Final) => {
            art!(6, [".#...", ".#...", ".###.", ".#.#.", "#####"])
        }

        // Ain: open jaw initial, closed eye medial, descender standalone.
        (Ain, Isolated) => art!(8, ["####", "#...", "####", "#...", "###."]),
        (Ain, Final) => art!(8, [".###.", ".#.#.", "#####", "#....", "####."]),
        (Ain, Initial) => art!(8, [".###", ".#..", "####"]),
        (Ain, Medial) => art!(8, [".###.", ".#.#.", "#####"]),

        // Feh: small eye + shallow bowl.
        (Feh, Isolated) => art!(6, ["....o.", "......", "#..###", "#..#.#", "######"]),
        (Feh, Final) => art!(6, ["....o..", ".......", "#..###.", "#..#.#.", "#######"]),
        (Feh, Initial) => art!(6, ["..o.", "....", ".###", ".#.#", "####"]),
        (Feh, Medial) => art!(6, ["..o..", ".....", ".###.", ".#.#.", "#####"]),

        // Qaf: small eye + deep descender bowl.
        (Qaf, Isolated) => art!(6, ["....o.", "......", "#..###", "#..#.#", "######", "#.....", "#####."]),
        (Qaf, Final) => art!(6, ["....o..", ".......", "#..###.", "#..#.#.", "#..####", "#......", "######."]),
        (Qaf, Initial) => art!(6, ["..o.", "....", ".###", ".#.#", "####"]),
        (Qaf, Medial) => art!(6, ["..o..", ".....", ".###.", ".#.#.", "#####"]),

        // Kaf: S-stroke with a tall stem.
        (Kaf, Isolated) => art!(6, ["...#", "...#", "####", "#...", "####"]),
        (Kaf, Initial) => art!(6, ["....#", "....#", ".####", ".#...", "#####"]),
        (Kaf, Medial) => art!(6, ["....#.", "....#.", ".####.", ".#....", "######"]),
        (Kaf, Final) => art!(6, ["...#.", "...#.", "####.", "#....", "#####"]),

        // Lam: tall vertical + baseline foot.
        (Lam, Isolated) => art!(2, ["...#", "...#", "...#", "...#", "...#", "...#", "...#", "...#", "####"]),
        (Lam, Initial) => art!(2, ["..#", "..#", "..#", "..#", "..#", "..#", "..#", "..#", "###"]),
        (Lam, Medial) => art!(2, ["..#.", "..#.", "..#.", "..#.", "..#.", "..#.", "..#.", "..#.", "####"]),
        (Lam, Final) => art!(2, ["...#.", "...#.", "...#.", "...#.", "...#.", "...#.", "...#.", "...#.", "#####"]),

        // Meem: eye on the baseline + straight descender tail standalone.
        (Meem, Isolated) => art!(8, [".###", ".#.#", "####", "#...", "#..."]),
        (Meem, Final) => art!(8, [".###.", ".#.#.", "#####", "#....", "#...."]),
        (Meem, Initial) => art!(8, [".###", ".#.#", "####"]),
        (Meem, Medial) => art!(8, [".###.", ".#.#.", "#####"]),

        // Heh: closed eye.
        (Heh, Isolated) => art!(8, ["###", "#.#", "###"]),
        (Heh, Final) => art!(8, ["###.", "#.#.", "####"]),
        (Heh, Initial) => art!(8, [".###", ".#.#", "####"]),
        (Heh, Medial) => art!(8, [".###.", ".#.#.", "#####"]),

        // Waw: eye + reh-like under-tail.
        (Waw, Isolated) => art!(8, [".###", ".#.#", "####", "#...", "##.."]),
        (Waw, Final) => art!(8, [".###.", ".#.#.", "#####", "#....", "##..."]),
        (Waw, _) => None,

        // Standalone hamza: small angular mark sitting on the baseline.
        (Hamza, Isolated) => art!(7, [".##", "..#", ".##", "##."]),
        (Hamza, _) => None,

        // Lam-alef ligature: the two stems joined at the baseline.
        (LamAlef, Isolated) => art!(2, ["#..#", "#..#", "#..#", "#..#", "#..#", "#..#", "#..#", "#..#", "####"]),
        (LamAlef, Final) => art!(2, ["#..#.", "#..#.", "#..#.", "#..#.", "#..#.", "#..#.", "#..#.", "#..#.", "#####"]),
        (LamAlef, _) => None,

        // Latin capitals: one form, cap height = alif height (rows 2–10).
        (Latin(i), Isolated) => Some(FormArt { top: 2, rows: latin_art(i).0 }),
        (Latin(_), _) => None,
    }
}

/// Latin capital art (rows 2–10, 1-cell strokes) and the letter's
/// stretch column: elongation duplicates that art column, so crossbars
/// and arms extend while stems stay one cell wide. None = the letter
/// does not stretch (diagonal or single-stem letters).
pub fn latin_art(i: u8) -> (&'static [&'static str], Option<usize>) {
    match i {
        0 /* A */ => (&["#####", "#...#", "#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"], Some(2)),
        1 /* B */ => (&["####.", "#...#", "#...#", "#...#", "####.", "#...#", "#...#", "#...#", "####."], Some(2)),
        2 /* C */ => (&["#####", "#....", "#....", "#....", "#....", "#....", "#....", "#....", "#####"], Some(2)),
        3 /* D */ => (&["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "#...#", "#...#", "####."], Some(2)),
        4 /* E */ => (&["#####", "#....", "#....", "#....", "#####", "#....", "#....", "#....", "#####"], Some(2)),
        5 /* F */ => (&["#####", "#....", "#....", "#....", "#####", "#....", "#....", "#....", "#...."], Some(2)),
        6 /* G */ => (&["#####", "#....", "#....", "#....", "#.###", "#...#", "#...#", "#...#", "#####"], Some(2)),
        7 /* H */ => (&["#...#", "#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#", "#...#"], Some(2)),
        8 /* I */ => (&["#", "#", "#", "#", "#", "#", "#", "#", "#"], None),
        9 /* J */ => (&["....#", "....#", "....#", "....#", "....#", "....#", "....#", "#...#", "#####"], Some(2)),
        10 /* K */ => (&["#...#", "#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#", "#...#"], None),
        11 /* L */ => (&["#....", "#....", "#....", "#....", "#....", "#....", "#....", "#....", "#####"], Some(2)),
        12 /* M */ => (&["#######", "#..#..#", "#..#..#", "#..#..#", "#.....#", "#.....#", "#.....#", "#.....#", "#.....#"], Some(1)),
        13 /* N */ => (&["#...#", "##..#", "##..#", "#.#.#", "#.#.#", "#.#.#", "#..##", "#..##", "#...#"], None),
        14 /* O */ => (&["#####", "#...#", "#...#", "#...#", "#...#", "#...#", "#...#", "#...#", "#####"], Some(2)),
        15 /* P */ => (&["#####", "#...#", "#...#", "#...#", "#####", "#....", "#....", "#....", "#...."], Some(2)),
        16 /* Q */ => (&["#####", "#...#", "#...#", "#...#", "#...#", "#...#", "#...#", "#..##", "#####"], Some(2)),
        17 /* R */ => (&["#####", "#...#", "#...#", "#####", "#.##.", "#..##", "#...#", "#...#", "#...#"], Some(1)),
        18 /* S */ => (&["#####", "#....", "#....", "#....", "#####", "....#", "....#", "....#", "#####"], Some(2)),
        19 /* T */ => (&["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."], Some(3)),
        20 /* U */ => (&["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", "#...#", "#...#", "#####"], Some(2)),
        21 /* V */ => (&["#...#", "#...#", "#...#", "##.##", ".#.#.", ".#.#.", ".###.", "..#..", "..#.."], None),
        22 /* W */ => (&["#.....#", "#.....#", "#.....#", "#.....#", "#.....#", "#..#..#", "#..#..#", "#..#..#", "#######"], Some(1)),
        23 /* X */ => (&["#...#", "#...#", "##.##", ".#.#.", ".###.", ".#.#.", "##.##", "#...#", "#...#"], None),
        24 /* Y */ => (&["#...#", "#...#", "#...#", "#...#", "#####", "..#..", "..#..", "..#..", "..#.."], Some(1)),
        _ /* Z */ => (&["#####", "....#", "...##", "...#.", "..##.", "..#..", ".##..", ".#...", "#####"], None),
    }
}

/// Render a glyph image from the teacher: class art, right-aligned on
/// the canvas, elongated per the class's stretch behaviour (Arabic:
/// kashida columns at the entry side; Latin: duplicated stretch
/// column), with i'jam dots stamped clear of the body.
pub fn render(class: Class, dots: Dots, form: Form, elong: usize) -> Option<GlyphImage> {
    let art = class_art(class, form)?;
    let base_w = art.rows.iter().map(|r| r.len()).max().unwrap_or(0);
    // Materialize the art so elongation can edit it. An `o` cell is a
    // dot anchor: it marks exactly where this form wants its i'jam
    // dots (designer-controlled), and renders empty.
    let mut anchor: Option<(usize, usize)> = None; // (art col, art row)
    let mut rows: Vec<Vec<bool>> = art
        .rows
        .iter()
        .enumerate()
        .map(|(dy, r)| {
            let mut v: Vec<bool> = r
                .bytes()
                .enumerate()
                .map(|(dx, b)| {
                    if b == b'o' {
                        anchor = Some((dx, dy));
                    }
                    b == b'#'
                })
                .collect();
            v.resize(base_w, false);
            v
        })
        .collect();
    // Where does the elongation go for this glyph?
    let (kashida, stretch_col) = match class {
        Class::Latin(i) => (0, latin_art(i).1.filter(|_| elong > 0)),
        _ if form.joins_prev() => (elong, None),
        _ => (0, None),
    };
    if let Some(col) = stretch_col {
        for row in &mut rows {
            let v = row[col];
            for _ in 0..elong {
                row.insert(col, v);
            }
        }
    }
    let aw = rows.iter().map(|r| r.len()).max().unwrap_or(0);

    let mut img = GlyphImage::empty();
    // Body right-aligned, shifted left by any kashida columns.
    let x0 = GRID_W - aw - kashida;
    for (dy, row) in rows.iter().enumerate() {
        for (dx, &on) in row.iter().enumerate() {
            if on {
                img.set(x0 + dx, art.top + dy);
            }
        }
    }
    // Kashida: extend the baseline through the elongation columns.
    for e in 0..kashida {
        img.set(GRID_W - 1 - e, BASELINE_ROW);
    }
    // Dots: at the form's explicit anchor when it has one, else placed
    // relative to the body bounding box.
    let anchor_abs = anchor.map(|(dx, dy)| (x0 + dx, art.top + dy));
    stamp_dots(&mut img, dots, x0, aw, art.top, art.top + rows.len() - 1, anchor_abs);
    img.advance = (aw + kashida) as f64;
    Some(img)
}

fn stamp_dots(
    img: &mut GlyphImage,
    dots: Dots,
    x0: usize,
    aw: usize,
    top: usize,
    bottom: usize,
    anchor: Option<(usize, usize)>,
) {
    if dots == 0 {
        return;
    }
    let n = dots.unsigned_abs() as usize;
    if let Some((ac, ar)) = anchor {
        let cells: Vec<(usize, usize)> = match n {
            1 => vec![(ac, ar)],
            2 => vec![(ac.saturating_sub(1), ar), (ac + 1, ar)],
            _ => vec![
                (ac.saturating_sub(1), ar),
                (ac + 1, ar),
                (ac, if dots > 0 { ar.saturating_sub(2) } else { ar + 2 }),
            ],
        };
        for (x, y) in cells {
            if y < GRID_H && x < GRID_W && clear_around(img, x, y) {
                img.set(x, y);
            }
        }
        return;
    }
    let c = (x0 + x0 + aw - 1) / 2; // body center column
    let row = if dots > 0 {
        if top < 2 {
            return;
        }
        top - 2
    } else {
        if bottom + 2 >= GRID_H {
            return; // no room below (e.g. descender forms) — omit, like ى
        }
        bottom + 2
    };
    let cells: Vec<(usize, usize)> = match n {
        1 => vec![(c, row)],
        2 => vec![(c.saturating_sub(1), row), (c + 1, row)],
        _ => vec![
            (c.saturating_sub(1), row),
            (c + 1, row),
            (c, if dots > 0 { row.saturating_sub(2) } else { row + 2 }),
        ],
    };
    for (x, y) in cells {
        if y < GRID_H && x < GRID_W && clear_around(img, x, y) {
            img.set(x, y);
        }
    }
}

/// A dot must not merge with the body: its cell and 4-neighbours must
/// be empty.
fn clear_around(img: &GlyphImage, x: usize, y: usize) -> bool {
    let nbrs = [(0i32, 0i32), (1, 0), (-1, 0), (0, 1), (0, -1)];
    nbrs.iter().all(|&(dx, dy)| {
        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
        nx < 0 || ny < 0 || !img.get(nx as usize, ny as usize)
    })
}
