#!/usr/bin/env python3
"""Data prep for the Nastaliq Distilled blog figures.

Composes words from the real training data (fields.bin + dataset.jsonl),
exactly like neuraltype-core/field_text.rs: chain displacements, subtract
the canvas origin, composite by max. Emits raw RGBA blobs that the
designbot scripts in this directory blit and annotate.

Usage: python3 figures.py <figure> <out.rgba>
Prints "W H" of the emitted blob on stdout.
"""
import json
import os
import sys

ROOT = os.path.expanduser("~/GH/repos/post-opentype/data/fields-gulzar-64")
META = json.load(open(f"{ROOT}/fields-meta.json"))
W, H = META["w"], META["h"]
OX, OY = META["origin_x"], META["origin_y"]
SCALE = META["em_px"] / META["upm"]  # font units -> px
BIN = open(f"{ROOT}/fields.bin", "rb").read()

GREEN = (42, 163, 95, 255)
GRAY = (110, 110, 110, 255)
RED = (239, 68, 68, 255)

ROWS = [json.loads(l) for l in open(f"{ROOT}/dataset.jsonl")]
BYFEAT = {}
for r in ROWS:
    key = (r["letters"], r["prev2"], r["prev"], r["next"], r["next2"])
    BYFEAT.setdefault(key, r)


def field(shape):
    return BIN[shape * W * H : (shape + 1) * W * H]


def clusters(word):
    """Cluster letters + 5-token features, with the لا fuse."""
    chars = list(word)
    cl = []
    i = 0
    while i < len(chars):
        if chars[i] == "ل" and i + 1 < len(chars) and chars[i + 1] == "ا":
            cl.append("لا")
            i += 2
        else:
            cl.append(chars[i])
            i += 1
    out = []
    for k, letters in enumerate(cl):
        get = lambda j: cl[j] if 0 <= j < len(cl) else None
        out.append((letters, (get(k - 2), get(k - 1), get(k + 1), get(k + 2))))
    return out


def compose(word):
    """-> (masks, w, h): one binary mask per cluster, in the word bbox."""
    recs = []
    ox = oy = 0.0
    for k, (letters, (p2, p, n, n2)) in enumerate(clusters(word)):
        r = BYFEAT[(letters, p2, p, n, n2)]
        if k == 0:
            ox, oy = r["ddx"], r["ddy"]
        else:
            ox += r["ddx"]
            oy += r["ddy"]
        recs.append((r["shape"], ox * SCALE, -oy * SCALE))
    x0 = min(px - OX for _, px, _ in recs)
    y0 = min(py - OY for _, _, py in recs)
    x1 = max(px - OX + W for _, px, _ in recs)
    y1 = max(py - OY + H for _, _, py in recs)
    w, h = int(x1 - x0 + 1), int(y1 - y0 + 1)
    masks = []
    for shape, px, py in recs:
        f = field(shape)
        m = bytearray(w * h)
        bx, by = round(px - OX - x0), round(py - OY - y0)
        for y in range(H):
            row = f[y * W : (y + 1) * W]
            ty = by + y
            for x in range(W):
                if row[x] >= 128:
                    m[ty * w + bx + x] = 1
        masks.append(m)
    return masks, w, h


def crop(masks, w, h, pad=6):
    on = [i for m in masks for i, v in enumerate(m) if v]
    xs = [i % w for i in on]
    ys = [i // w for i in on]
    x0, x1 = max(min(xs) - pad, 0), min(max(xs) + pad, w - 1)
    y0, y1 = max(min(ys) - pad, 0), min(max(ys) + pad, h - 1)
    cw, ch = x1 - x0 + 1, y1 - y0 + 1
    out = []
    for m in masks:
        c = bytearray(cw * ch)
        for y in range(ch):
            for x in range(cw):
                c[y * cw + x] = m[(y0 + y) * w + x0 + x]
        out.append(c)
    return out, cw, ch


def blit(buf, bw, bh, mask, mw, mh, x0, y0, color):
    for y in range(mh):
        for x in range(mw):
            if mask[y * mw + x]:
                i = ((y0 + y) * bw + x0 + x) * 4
                buf[i : i + 4] = bytes(color)


def fig_noon_variants():
    """The 16 first-position shapes of ن, each in the pair that selects
    it. ن in green, the follower in gray."""
    seen = {}
    for r in ROWS:
        if (
            r["letters"] == "ن"
            and r["prev"] is None
            and r["prev2"] is None
            and r["next2"] is None
            and r["next"] is not None
        ):
            seen.setdefault(r["shape"], r["next"])
    pairs = ["ن" + n for n in seen.values()]
    cells = []
    for p in pairs:
        masks, w, h = compose(p)
        masks, w, h = crop(masks, w, h)
        cells.append((masks, w, h))
    cols = 7
    rows = (len(cells) + cols - 1) // cols
    cw = max(c[1] for c in cells)
    ch = max(c[2] for c in cells)
    pad = 14
    bw = cols * (cw + pad) + pad
    bh = rows * (ch + pad) + pad
    buf = bytearray(bw * bh * 4)
    for k, (masks, w, h) in enumerate(cells):
        cx = pad + (k % cols) * (cw + pad) + (cw - w) // 2
        cy = pad + (k // cols) * (ch + pad) + (ch - h) // 2
        blit(buf, bw, bh, masks[1], w, h, cx, cy, GRAY)
        blit(buf, bw, bh, masks[0], w, h, cx, cy, GREEN)
    return buf, bw, bh


def fig_kha_sheet():
    """Every distinct composed shape of خ in the dataset, one small
    green cell each."""
    shapes = sorted({r["shape"] for r in ROWS if r["letters"] == "خ"})
    cells = []
    for sh in shapes:
        m = bytearray(1 if v >= 128 else 0 for v in field(sh))
        c, cw, ch = crop([m], W, H, pad=3)
        cells.append((c[0], cw, ch))
    cols = 13
    rows = (len(cells) + cols - 1) // cols
    cw = max(c[1] for c in cells)
    ch = max(c[2] for c in cells)
    pad = 10
    bw = cols * (cw + pad) + pad
    bh = rows * (ch + pad) + pad
    buf = bytearray(bw * bh * 4)
    for k, (m, w, h) in enumerate(cells):
        cx = pad + (k % cols) * (cw + pad) + (cw - w) // 2
        cy = pad + (k // cols) * (ch + pad) + (ch - h) // 2
        blit(buf, bw, bh, m, w, h, cx, cy, GREEN)
    sys.stderr.write(f"kha shapes: {len(cells)}\n")
    return buf, bw, bh


def fig_res():
    """The isolated خ thresholded at 64 px/em (left) and 96 px/em
    (right), nearest-neighbor upscaled to the same physical size."""
    panels = []
    for em, mult in ((64, 6), (96, 4)):
        root = os.path.expanduser(f"~/GH/repos/post-opentype/data/fields-gulzar-{em}")
        meta = json.load(open(f"{root}/fields-meta.json"))
        w, h = meta["w"], meta["h"]
        rows = [json.loads(l) for l in open(f"{root}/dataset.jsonl")]
        sh = next(
            r["shape"]
            for r in rows
            if r["letters"] == "خ" and not any(r.get(k) for k in ("prev", "prev2", "next", "next2"))
        )
        data = open(f"{root}/fields.bin", "rb").read()
        f = data[sh * w * h : (sh + 1) * w * h]
        m = bytearray(1 if v >= 128 else 0 for v in f)
        ms, cw, ch = crop([m], w, h, pad=3)
        m = ms[0]
        # nearest-neighbor upscale by mult
        uw, uh = cw * mult, ch * mult
        u = bytearray(uw * uh)
        for y in range(uh):
            for x in range(uw):
                u[y * uw + x] = m[(y // mult) * cw + x // mult]
        panels.append((u, uw, uh))
    pad = 60
    bh = max(p[2] for p in panels) + 2 * pad
    bw = sum(p[1] for p in panels) + 3 * pad
    buf = bytearray(bw * bh * 4)
    x = pad
    for u, uw, uh in panels:
        blit(buf, bw, bh, u, uw, uh, x, (bh - uh) // 2, GREEN)
        x += uw + pad
    return buf, bw, bh


FIGS = {
    "noon-variants": fig_noon_variants,
    "kha-sheet": fig_kha_sheet,
    "res": fig_res,
}

if __name__ == "__main__":
    fig, out = sys.argv[1], sys.argv[2]
    buf, w, h = FIGS[fig]()
    open(out, "wb").write(bytes(buf))
    print(w, h)
