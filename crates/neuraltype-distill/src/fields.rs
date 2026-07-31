//! Fields: render extracted cluster shapes to signed-distance fields.
//!
//! Step 2 of the TTF→NTF conversion (docs/DISTILL.md). Every unique
//! composed cluster shape (base glyph plus marks) is rasterized at 4×
//! supersample, converted to an exact Euclidean signed-distance field,
//! clamped, and stored as one u8 grid per shape. All shapes share one
//! canvas with the cluster origin at a fixed anchor pixel, so fields
//! composite by pointwise max at chained origins, exactly as v0
//! composited bitmaps.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write as _;

#[derive(Deserialize)]
struct GlyphRec {
    gid: u32,
    path: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PlacedGlyph {
    pub gid: u32,
    pub dx: i32,
    pub dy: i32,
}

#[derive(Deserialize)]
struct ClusterRec {
    letters: String,
    index: usize,
    prev: Option<char>,
    next: Option<char>,
    #[serde(default)]
    prev2: Option<char>,
    #[serde(default)]
    next2: Option<char>,
    glyphs: Vec<PlacedGlyph>,
    ddx: Option<i32>,
    ddy: Option<i32>,
}

#[derive(Serialize)]
struct FieldsMeta {
    em_px: u32,
    upm: f64,
    /// Canvas size in pixels.
    w: u32,
    h: u32,
    /// The cluster origin, in pixels from the canvas top-left.
    origin_x: f64,
    origin_y: f64,
    /// SDF clamp distance in pixels; u8 128 = on the contour.
    spread_px: f64,
    shapes: usize,
}

/// Parse the extract stage's SVG-ish path into a kurbo BezPath.
fn parse_path(d: &str) -> kurbo::BezPath {
    let mut p = kurbo::BezPath::new();
    let toks: Vec<&str> = d
        .split_inclusive(|c: char| c.is_ascii_alphabetic())
        .flat_map(|s| s.split_whitespace())
        .collect();
    // Simpler: tokenize by regex-free scan.
    let mut nums: Vec<f64> = Vec::new();
    let mut cmd = ' ';
    let mut cur = String::new();
    let mut flush_num = |cur: &mut String, nums: &mut Vec<f64>| {
        if !cur.is_empty() {
            nums.push(cur.parse().unwrap());
            cur.clear();
        }
    };
    let mut apply = |cmd: char, nums: &mut Vec<f64>, p: &mut kurbo::BezPath| {
        match cmd {
            'M' => p.move_to((nums[0], nums[1])),
            'L' => p.line_to((nums[0], nums[1])),
            'Q' => p.quad_to((nums[0], nums[1]), (nums[2], nums[3])),
            'C' => p.curve_to((nums[0], nums[1]), (nums[2], nums[3]), (nums[4], nums[5])),
            'Z' => p.close_path(),
            _ => {}
        }
        nums.clear();
    };
    let _ = toks;
    for ch in d.chars() {
        match ch {
            'M' | 'L' | 'Q' | 'C' | 'Z' => {
                flush_num(&mut cur, &mut nums);
                if cmd != ' ' {
                    apply(cmd, &mut nums, &mut p);
                }
                cmd = ch;
                if ch == 'Z' {
                    apply('Z', &mut nums, &mut p);
                    cmd = ' ';
                }
            }
            ' ' => flush_num(&mut cur, &mut nums),
            _ => cur.push(ch),
        }
    }
    flush_num(&mut cur, &mut nums);
    if cmd != ' ' {
        apply(cmd, &mut nums, &mut p);
    }
    p
}

/// Fill a path (nonzero winding) into a supersampled binary grid.
/// The transform maps font units to pixel space (y flipped).
fn rasterize(
    paths: &[(kurbo::BezPath, f64, f64)], // path, dx, dy in font units
    w: usize,
    h: usize,
    scale: f64,       // px per font unit (already includes supersample)
    off_x: f64,       // font-unit x of pixel column 0
    off_y_top: f64,   // font-unit y of pixel row 0 (top)
) -> Vec<bool> {
    // Flatten all paths to line segments in pixel space.
    let mut segs: Vec<(f64, f64, f64, f64)> = Vec::new();
    for (path, dx, dy) in paths {
        let a = kurbo::Affine::translate((*dx, *dy));
        let moved = a * path.clone();
        let mut last = kurbo::Point::ZERO;
        let mut start = kurbo::Point::ZERO;
        moved.flatten(0.1, |el| match el {
            kurbo::PathEl::MoveTo(pt) => {
                last = pt;
                start = pt;
            }
            kurbo::PathEl::LineTo(pt) => {
                segs.push((last.x, last.y, pt.x, pt.y));
                last = pt;
            }
            kurbo::PathEl::ClosePath => {
                segs.push((last.x, last.y, start.x, start.y));
                last = start;
            }
            _ => {}
        });
    }
    // To pixel space: px = (x - off_x) * scale, py = (off_y_top - y) * scale.
    let segs: Vec<(f64, f64, f64, f64)> = segs
        .iter()
        .map(|(x0, y0, x1, y1)| {
            (
                (x0 - off_x) * scale,
                (off_y_top - y0) * scale,
                (x1 - off_x) * scale,
                (off_y_top - y1) * scale,
            )
        })
        .collect();

    let mut grid = vec![false; w * h];
    for row in 0..h {
        let py = row as f64 + 0.5;
        // crossings with winding direction
        let mut xs: Vec<(f64, i32)> = Vec::new();
        for &(x0, y0, x1, y1) in &segs {
            if (y0 <= py && y1 > py) || (y1 <= py && y0 > py) {
                let t = (py - y0) / (y1 - y0);
                let x = x0 + t * (x1 - x0);
                xs.push((x, if y1 > y0 { 1 } else { -1 }));
            }
        }
        xs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut winding = 0;
        let mut i = 0;
        for col in 0..w {
            let px = col as f64 + 0.5;
            while i < xs.len() && xs[i].0 <= px {
                winding += xs[i].1;
                i += 1;
            }
            grid[row * w + col] = winding != 0;
        }
    }
    grid
}

/// Exact 1D squared-distance transform (Felzenszwalb & Huttenlocher).
fn dt1d(f: &[f64], out: &mut [f64]) {
    let n = f.len();
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        loop {
            let s = ((f[q] + (q * q) as f64) - (f[v[k]] + (v[k] * v[k]) as f64))
                / (2.0 * q as f64 - 2.0 * v[k] as f64);
            if s <= z[k] {
                if k == 0 {
                    // shouldn't happen with -inf sentinel, but be safe
                    break;
                }
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
                break;
            }
        }
    }
    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let d = q as f64 - v[k] as f64;
        out[q] = d * d + f[v[k]];
    }
}

/// Exact Euclidean distance transform of a binary grid: distance from
/// every cell to the nearest `true` cell.
fn edt(grid: &[bool], w: usize, h: usize) -> Vec<f64> {
    const INF: f64 = 1e18;
    let mut d: Vec<f64> = grid.iter().map(|&b| if b { 0.0 } else { INF }).collect();
    // columns
    let mut col = vec![0.0f64; h];
    let mut out = vec![0.0f64; h];
    for x in 0..w {
        for y in 0..h {
            col[y] = d[y * w + x];
        }
        dt1d(&col, &mut out);
        for y in 0..h {
            d[y * w + x] = out[y];
        }
    }
    // rows
    let mut row = vec![0.0f64; w];
    let mut out = vec![0.0f64; w];
    for y in 0..h {
        row.copy_from_slice(&d[y * w..(y + 1) * w]);
        dt1d(&row, &mut out);
        d[y * w..(y + 1) * w].copy_from_slice(&out);
    }
    d.iter_mut().for_each(|v| *v = v.sqrt());
    d
}

pub fn fields(extract_dir: &str, out_dir: &str, em_px: u32) {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{extract_dir}/meta.json")).unwrap(),
    )
    .unwrap();
    let upm = meta["upm"].as_f64().unwrap();
    std::fs::create_dir_all(out_dir).unwrap();

    // Glyph outlines.
    let mut outlines: HashMap<u32, kurbo::BezPath> = HashMap::new();
    for line in std::fs::read_to_string(format!("{extract_dir}/glyphs.jsonl"))
        .unwrap()
        .lines()
    {
        let g: GlyphRec = serde_json::from_str(line).unwrap();
        outlines.insert(g.gid, parse_path(&g.path));
    }

    // Dedupe shapes; build the context dataset referencing shape ids.
    let mut shape_ids: HashMap<String, usize> = HashMap::new();
    let mut shapes: Vec<Vec<PlacedGlyph>> = Vec::new();
    let ctx_out =
        std::fs::File::create(format!("{out_dir}/dataset.jsonl")).unwrap();
    let mut ctx_w = std::io::BufWriter::new(ctx_out);
    let contexts = std::fs::read_to_string(format!("{extract_dir}/contexts.jsonl")).unwrap();
    let mut n_rows = 0usize;
    for line in contexts.lines() {
        let r: ClusterRec = serde_json::from_str(line).unwrap();
        let key = serde_json::to_string(&r.glyphs).unwrap();
        let id = *shape_ids.entry(key).or_insert_with(|| {
            shapes.push(r.glyphs.clone());
            shapes.len() - 1
        });
        writeln!(
            ctx_w,
            "{}",
            serde_json::json!({
                "letters": r.letters, "prev": r.prev, "next": r.next,
                "prev2": r.prev2, "next2": r.next2,
                "index": r.index, "shape": id, "ddx": r.ddx, "ddy": r.ddy,
            })
        )
        .unwrap();
        n_rows += 1;
    }
    println!("{n_rows} context rows, {} unique shapes", shapes.len());

    // Global bbox over all shapes, relative to the cluster origin.
    let mut bb = kurbo::Rect::new(0.0, 0.0, 0.0, 0.0);
    let mut first = true;
    for shape in &shapes {
        for pg in shape {
            if let Some(path) = outlines.get(&pg.gid) {
                if path.elements().is_empty() {
                    continue;
                }
                let r = kurbo::Shape::bounding_box(path)
                    + kurbo::Vec2::new(pg.dx as f64, pg.dy as f64);
                bb = if first { r } else { bb.union(r) };
                first = false;
            }
        }
    }
    println!(
        "shape bbox (font units, rel. origin): x {:.0}..{:.0}  y {:.0}..{:.0}",
        bb.x0, bb.x1, bb.y0, bb.y1
    );

    let px_per_unit = em_px as f64 / upm;
    let spread_px = em_px as f64 * 0.125; // clamp at 1/8 em
    let pad = spread_px / px_per_unit; // font units of padding
    let x0 = bb.x0 - pad;
    let x1 = bb.x1 + pad;
    let y0 = bb.y0 - pad;
    let y1 = bb.y1 + pad;
    let w = ((x1 - x0) * px_per_unit).ceil() as usize;
    let h = ((y1 - y0) * px_per_unit).ceil() as usize;
    println!("canvas: {w}×{h} px at {em_px} px/em ({} shapes)", shapes.len());

    const SS: usize = 4; // supersample factor
    let sw = w * SS;
    let sh = h * SS;
    let mut fields_bin: Vec<u8> = Vec::with_capacity(shapes.len() * w * h);

    for (si, shape) in shapes.iter().enumerate() {
        if si % 200 == 0 {
            println!("  field {si}/{}", shapes.len());
        }
        let paths: Vec<(kurbo::BezPath, f64, f64)> = shape
            .iter()
            .filter_map(|pg| {
                outlines
                    .get(&pg.gid)
                    .map(|p| (p.clone(), pg.dx as f64, pg.dy as f64))
            })
            .collect();
        let grid = rasterize(&paths, sw, sh, px_per_unit * SS as f64, x0, y1);
        // signed distance: outside dist − inside dist, in target pixels
        let inv: Vec<bool> = grid.iter().map(|b| !b).collect();
        let d_out = edt(&grid, sw, sh); // distance to figure (for ground cells)
        let d_in = edt(&inv, sw, sh); // distance to ground (for figure cells)
        for row in 0..h {
            for col in 0..w {
                // sample the supersampled center
                let sy = row * SS + SS / 2;
                let sx = col * SS + SS / 2;
                let i = sy * sw + sx;
                let sd = if grid[i] { d_in[i] } else { -d_out[i] } / SS as f64;
                let v = (sd / spread_px).clamp(-1.0, 1.0);
                fields_bin.push(((v * 127.0) + 128.0) as u8);
            }
        }
    }

    std::fs::write(format!("{out_dir}/fields.bin"), &fields_bin).unwrap();
    let shapes_json: Vec<serde_json::Value> = shapes
        .iter()
        .map(|s| serde_json::to_value(s).unwrap())
        .collect();
    std::fs::write(
        format!("{out_dir}/shapes.json"),
        serde_json::to_string(&shapes_json).unwrap(),
    )
    .unwrap();
    let fm = FieldsMeta {
        em_px,
        upm,
        w: w as u32,
        h: h as u32,
        origin_x: (0.0 - x0) * px_per_unit,
        origin_y: (y1 - 0.0) * px_per_unit,
        spread_px,
        shapes: shapes.len(),
    };
    std::fs::write(
        format!("{out_dir}/fields-meta.json"),
        serde_json::to_string_pretty(&fm).unwrap(),
    )
    .unwrap();
    println!(
        "wrote {} fields ({} bytes) to {out_dir}",
        shapes.len(),
        fields_bin.len()
    );
}

/// Write a proof sheet of the first/selected fields as a PGM image
/// (convert to PNG with `magick sheet.pgm sheet.png`).
pub fn proof(fields_dir: &str, out_path: &str, ids: &[usize]) {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{fields_dir}/fields-meta.json")).unwrap(),
    )
    .unwrap();
    let (w, h) = (
        meta["w"].as_u64().unwrap() as usize,
        meta["h"].as_u64().unwrap() as usize,
    );
    let n = meta["shapes"].as_u64().unwrap() as usize;
    let bin = std::fs::read(format!("{fields_dir}/fields.bin")).unwrap();
    let ids: Vec<usize> = if ids.is_empty() {
        (0..n.min(24)).collect()
    } else {
        ids.to_vec()
    };
    let cols = 6usize.min(ids.len());
    let rows = ids.len().div_ceil(cols);
    let gap = 4usize;
    let sheet_w = cols * (w + gap);
    let sheet_h = rows * (h + gap);
    let mut sheet = vec![0u8; sheet_w * sheet_h];
    for (k, &id) in ids.iter().enumerate() {
        let (r, c) = (k / cols, k % cols);
        for y in 0..h {
            for x in 0..w {
                let v = bin[id * w * h + y * w + x];
                // show the figure: map sdf>=128 (inside) bright,
                // near-contour gray, ground dark
                sheet[(r * (h + gap) + y) * sheet_w + c * (w + gap) + x] = v;
            }
        }
    }
    let mut f = std::fs::File::create(out_path).unwrap();
    writeln!(f, "P5\n{sheet_w} {sheet_h}\n255").unwrap();
    f.write_all(&sheet).unwrap();
    println!("wrote {out_path} ({} shapes)", ids.len());
}
