#!/usr/bin/env python3
"""Build a three-row comparison sheet: teacher, field model, vector model.

Every row draws the same words in one coordinate frame (font units,
y-up, baseline at 0), so differences are shape differences, not
placement or scale artifacts.

  teacher  composed from the extraction (Gulzar's own outlines)
  field    the field .ntf, img2bez-traced (what the demo draws)
  vector   the vector .ntf, greedy-decoded outline tokens

Usage:
  python3 scripts/sample_sheet.py <field.ntf> <vector.ntf> <out.html> [words...]
"""
import json
import os
import re
import subprocess
import sys
from datetime import date

ROOT = os.path.expanduser("~/GH/repos/post-opentype")
EXTRACT = f"{ROOT}/data/extract-gulzar"
DISTILL = f"{ROOT}/target/release/distill"
VEC = f"{ROOT}/target/release/ntf-train-vec"
INK = "#2aa35f"
DATE = date.today().strftime("%d %B %Y")

DEFAULT_WORDS = ["بسم", "الله", "نور", "سلام", "قلم", "نستعليق"]


def teacher_paths():
    """gid -> absolute SVG path string, from the extraction."""
    out = {}
    for line in open(f"{EXTRACT}/glyphs.jsonl"):
        r = json.loads(line)
        out[r["gid"]] = r["path"] or ""
    return out


def teacher_word(word, glyphs, contexts):
    """Compose one word's teacher outline: every cluster's glyphs at
    their recorded offsets, chained by the cluster origins."""
    recs = contexts.get(word)
    if not recs:
        return None
    d = []
    for r in recs:
        for g in r["glyphs"]:
            p = glyphs.get(g["gid"], "")
            if not p:
                continue
            d.append(translate(p, g["dx"] + r["ox"], g["dy"] + r["oy"]))
    return " ".join(d)


def translate(path_d, dx, dy):
    """Shift an absolute M/L/Q/C/Z path by (dx, dy)."""
    toks = re.findall(r"[MLQCZ]|-?\d+\.?\d*(?:e-?\d+)?", path_d)
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]
        i += 1
        n = {"M": 2, "L": 2, "Q": 4, "C": 6, "Z": 0}[t]
        out.append(t)
        for k in range(0, n, 2):
            x = float(toks[i + k]) + dx
            y = float(toks[i + k + 1]) + dy
            out.append(f"{x:.1f} {y:.1f}")
        i += n
    return " ".join(out)


def contexts_by_word():
    """word -> its cluster records, in order."""
    by = {}
    for line in open(f"{EXTRACT}/contexts.jsonl"):
        r = json.loads(line)
        by.setdefault(r["word"], []).append(r)
    for recs in by.values():
        recs.sort(key=lambda r: r["index"])
    return by


def ntf_format(ntf):
    """The header's format string, so the right binary decodes it."""
    with open(ntf, "rb") as f:
        head = f.read(8)
        hlen = int.from_bytes(head[4:8], "little")
        return json.loads(f.read(hlen)).get("format", "")


def model_word(ntf, word):
    """Run a model's wordjson and convert to font units, y-up. The
    field and vector formats live in different binaries."""
    binary = VEC if "vector" in ntf_format(ntf) else DISTILL
    args = [binary, "wordjson", ntf, word]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=600)
        if out.returncode != 0:
            return None
        r = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return None
    # px (y-down, baseline 0) -> font units (y-up)
    k = r["upm"] / r["em_px"]
    return svg_scale(r["d"], k, -k)


def svg_scale(path_d, sx, sy):
    toks = re.findall(r"[MLQCZ]|-?\d+\.?\d*(?:e-?\d+)?", path_d)
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]
        i += 1
        n = {"M": 2, "L": 2, "Q": 4, "C": 6, "Z": 0}[t]
        out.append(t)
        for k in range(0, n, 2):
            out.append(f"{float(toks[i + k]) * sx:.1f} {float(toks[i + k + 1]) * sy:.1f}")
        i += n
    return " ".join(out)


def bbox(path_d):
    nums = [float(v) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", path_d or "")]
    if not nums:
        return None
    xs, ys = nums[0::2], nums[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def union(boxes):
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def contours(path_d):
    return (path_d or "").count("M")


def cell(path_d, vb, note):
    """One plate. Every plate in a row shares `vb`, so a fragmented
    output looks small instead of being auto-zoomed to fill."""
    if not path_d:
        return (
            f'<div class="plate"><div class="frame empty">'
            f'<span>{note}</span></div></div>'
        )
    body = (
        f'<svg viewBox="{vb}" preserveAspectRatio="xMidYMid meet" role="img">'
        f'<g transform="scale(1,-1)"><path d="{path_d}" fill="var(--ink)"/></g></svg>'
    )
    n = contours(path_d)
    return (
        f'<div class="plate"><div class="frame">{body}</div>'
        f'<div class="meta">{n} contour{"" if n == 1 else "s"}</div></div>'
    )


def describe(ntf):
    """Facts a sheet can state without being told: the model's own
    weight count and the file size on disk."""
    size = os.path.getsize(ntf)
    with open(ntf, "rb") as f:
        hlen = int.from_bytes(f.read(8)[4:8], "little")
    params = (size - 8 - hlen) // 4
    return params, size


def main():
    field_ntf = sys.argv[1]
    vec_ntf = sys.argv[2]
    out_path = sys.argv[3]
    words = sys.argv[4:] or DEFAULT_WORDS
    # Labels and the closing note come from the caller: the same
    # script builds field-vs-vector and narrow-vs-wide sheets.
    title = os.environ.get("SHEET_TITLE", "Two models, one nastaliq")
    lede = os.environ.get("SHEET_LEDE", "")
    verdict = os.environ.get("SHEET_VERDICT", "")
    name_b = os.environ.get("SHEET_B", "model B")
    name_c = os.environ.get("SHEET_C", "model C")
    note_b = os.environ.get("SHEET_B_NOTE", "")
    note_c = os.environ.get("SHEET_C_NOTE", "")
    score_b = os.environ.get("SHEET_B_SCORE", "")
    score_c = os.environ.get("SHEET_C_SCORE", "")
    pb, sb = describe(field_ntf)
    pc, sc = describe(vec_ntf)

    glyphs = teacher_paths()
    ctx = contexts_by_word()
    rows = []
    for w in words:
        t = teacher_word(w, glyphs, ctx)
        f = model_word(field_ntf, w)
        v = model_word(vec_ntf, w)
        rows.append((w, t, f, v))
        print(f"{w}: teacher {'ok' if t else '--'}  field {'ok' if f else '--'}  "
              f"vector {'ok' if v else '--'}", flush=True)

    grid = ['<div class="grid">',
            '<div class="colhead"></div>',
            '<div class="colhead">teacher <em>Gulzar outlines</em></div>',
            f'<div class="colhead">{name_b} <em>{note_b}</em></div>',
            f'<div class="colhead">{name_c} <em>{note_c}</em></div>']
    for w, t, f, v in rows:
        b = union([bbox(t), bbox(f), bbox(v)])
        pad = 120
        if b:
            x0, y0, x1, y1 = b
            vb = f"{x0 - pad} {-(y1 + pad)} {x1 - x0 + 2 * pad} {y1 - y0 + 2 * pad}"
        else:
            vb = "0 0 1000 1000"
        grid.append(f'<div class="word" lang="ar">{w}</div>')
        grid.append(cell(t, vb, "not in the training corpus"))
        grid.append(cell(f, vb, "no output"))
        grid.append(cell(v, vb, "no output"))
    grid.append("</div>")

    page = f"""<title>NeuralType sample sheet</title>
<style>
:root {{
  --ink: #2aa35f;
  --ground: #f4f7f2;
  --surface: #ffffff;
  --line: #dde3d9;
  --text: #171a16;
  --muted: #5c635a;
  --flag: #b0503f;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --ground: #0d0f0d; --surface: #161916; --line: #252a25;
    --text: #e2e6e0; --muted: #8f968d; --flag: #d4776a;
  }}
}}
:root[data-theme="dark"] {{
  --ground: #0d0f0d; --surface: #161916; --line: #252a25;
  --text: #e2e6e0; --muted: #8f968d; --flag: #d4776a;
}}
:root[data-theme="light"] {{
  --ground: #f4f7f2; --surface: #ffffff; --line: #dde3d9;
  --text: #171a16; --muted: #5c635a; --flag: #b0503f;
}}
body {{ margin: 0; padding: 40px 28px 72px; background: var(--ground);
        color: var(--text); font-family: var(--mono); }}
.wrap {{ max-width: 1180px; margin: 0 auto; display: flex;
         flex-direction: column; gap: 34px; }}
.eyebrow {{ font-size: 11.5px; letter-spacing: 0.14em; text-transform: uppercase;
            color: var(--muted); }}
h1 {{ font-family: var(--serif); font-size: 30px; font-weight: 600; margin: 6px 0 0;
      letter-spacing: -0.01em; text-wrap: balance; }}
.lede {{ font-family: var(--serif); font-size: 16.5px; line-height: 1.62;
         color: var(--muted); max-width: 64ch; margin: 12px 0 0; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
          gap: 12px; }}
.stat {{ background: var(--surface); border: 1px solid var(--line);
         padding: 16px 18px; display: flex; flex-direction: column; gap: 7px; }}
.stat h2 {{ font-size: 12px; font-weight: 600; margin: 0; letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--text); }}
.stat dl {{ margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 3px 12px;
            font-size: 12.5px; }}
.stat dt {{ color: var(--muted); }}
.stat dd {{ margin: 0; text-align: right; font-variant-numeric: tabular-nums; }}
.chip {{ align-self: flex-start; font-size: 11px; letter-spacing: 0.05em;
         padding: 3px 8px; border: 1px solid currentColor; }}
.chip.good {{ color: var(--ink); }}
.chip.bad {{ color: var(--flag); }}
.scroller {{ overflow-x: auto; }}
.grid {{ display: grid; grid-template-columns: 92px repeat(3, minmax(210px, 1fr));
         gap: 12px; align-items: center; min-width: 760px; }}
.colhead {{ font-size: 12px; color: var(--text); text-align: center;
            display: flex; flex-direction: column; gap: 3px; }}
.colhead em {{ font-style: normal; font-size: 11px; color: var(--muted); }}
.word {{ font-size: 24px; text-align: right; padding-right: 6px; direction: rtl; }}
.plate {{ display: flex; flex-direction: column; gap: 5px; }}
.frame {{ background: var(--surface); border: 1px solid var(--line); height: 168px;
          display: flex; align-items: center; justify-content: center; }}
.frame svg {{ width: 100%; height: 100%; }}
.frame.empty span {{ font-size: 11px; color: var(--muted); }}
.meta {{ font-size: 10.5px; color: var(--muted); text-align: center;
         font-variant-numeric: tabular-nums; }}
.verdict {{ background: var(--surface); border: 1px solid var(--line);
            border-top: 2px solid var(--ink); padding: 22px 24px;
            display: flex; flex-direction: column; gap: 12px; }}
.verdict p {{ font-family: var(--serif); font-size: 15.5px; line-height: 1.65;
              margin: 0; max-width: 70ch; color: var(--text); }}
.verdict p + p {{ color: var(--muted); }}
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">NeuralType · {DATE}</div>
    <h1>{title}</h1>
    <p class="lede">{lede}</p>
  </header>

  <section class="stats">
    <div class="stat">
      <h2>Teacher</h2>
      <dl><dt>source</dt><dd>Gulzar OFL</dd>
      <dt>outlines</dt><dd>hand-drawn</dd></dl>
      <span class="chip good">ground truth</span>
    </div>
    <div class="stat">
      <h2>{name_b}</h2>
      <dl><dt>parameters</dt><dd>{pb:,}</dd>
      <dt>file size</dt><dd>{sb / 1e6:.1f} MB</dd>
      {f"<dt>contour IoU</dt><dd>{score_b}</dd>" if score_b else ""}</dl>
    </div>
    <div class="stat">
      <h2>{name_c}</h2>
      <dl><dt>parameters</dt><dd>{pc:,}</dd>
      <dt>file size</dt><dd>{sc / 1e6:.1f} MB</dd>
      {f"<dt>contour IoU</dt><dd>{score_c}</dd>" if score_c else ""}</dl>
    </div>
  </section>

  <div class="scroller">
    {chr(10).join(grid)}
  </div>

  <section class="verdict">{verdict}</section>
</div>
"""
    open(out_path, "w").write(page)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
