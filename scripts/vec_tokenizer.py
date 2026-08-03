#!/usr/bin/env python3
"""Vector-track tokenizer study (exploratory; see docs/VECTOR.md).

Reads the teacher's outlines (data/extract-gulzar/glyphs.jsonl) and
the cluster records (contexts.jsonl), composes each unique
letter-in-context cluster into one outline point sequence, and
tokenizes it Virtua-style: command tokens plus quantized coordinate
deltas, origin-relative.

Token stream per cluster:
  BOS, then per subpath: M dx dy, then (L dx dy | Q dx dy dx dy |
  C dx dy dx dy dx dy)*, Z, ... then EOS.
  The first M is relative to the cluster origin; every later
  coordinate is a delta from the previous point. Deltas outside the
  clamp range split into repeated MAX-step tokens (counted).

Outputs:
  data/vec-gulzar/vocab.json        token id table
  data/vec-gulzar/sequences.jsonl   {feats, tokens} per unique context
  data/vec-gulzar/meta.json         stats the decoder dims come from
Prints the study numbers.

Usage: python3 scripts/vec_tokenizer.py [q]   (q = units per bin, default 4)
"""
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.expanduser("~/GH/repos/post-opentype/data/extract-gulzar")
OUT = os.path.expanduser("~/GH/repos/post-opentype/data/vec-gulzar")
Q = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0  # font units per bin
DMAX = 63  # max |delta| in bins per token; larger deltas chain


def parse_path(d):
    """SVG path -> list of subpaths; each is (cmd, points) list.
    Handles M L Q C Z with absolute coords, as written by extraction."""
    import re
    tokens = re.findall(r"[MLQCZ]|-?\d+\.?\d*(?:e-?\d+)?", d)
    subpaths = []
    cur = None
    i = 0
    def take(n):
        nonlocal i
        vals = [float(tokens[i + k]) for k in range(n)]
        i += n
        return vals
    while i < len(tokens):
        t = tokens[i]
        i += 1
        if t == "M":
            if cur:
                subpaths.append(cur)
            x, y = take(2)
            cur = [("M", [(x, y)])]
        elif t == "L":
            x, y = take(2)
            cur.append(("L", [(x, y)]))
        elif t == "Q":
            x1, y1, x, y = take(4)
            cur.append(("Q", [(x1, y1), (x, y)]))
        elif t == "C":
            x1, y1, x2, y2, x, y = take(6)
            cur.append(("C", [(x1, y1), (x2, y2), (x, y)]))
        elif t == "Z":
            if cur:
                subpaths.append(cur)
                cur = None
    if cur:
        subpaths.append(cur)
    return subpaths


def main():
    glyph_paths = {}
    for line in open(f"{ROOT}/glyphs.jsonl"):
        r = json.loads(line)
        glyph_paths[r["gid"]] = parse_path(r["path"]) if r["path"] else []

    # unique contexts: first record per feature tuple (the corpus is
    # unambiguous at 5 tokens, so first == only)
    by_feat = {}
    for line in open(f"{ROOT}/contexts.jsonl"):
        r = json.loads(line)
        key = (r["letters"], r.get("prev2"), r.get("prev"), r.get("next"), r.get("next2"))
        by_feat.setdefault(key, r)

    # vocabulary
    vocab = ["<pad>", "<bos>", "<eos>", "M", "L", "Q", "C", "Z"]
    delta_base = len(vocab)
    for d in range(-DMAX, DMAX + 1):
        vocab.append(f"d{d}")
    tok = {t: i for i, t in enumerate(vocab)}

    def emit_delta(out, bins, stats):
        """One axis delta in bins -> one or more chained tokens."""
        while True:
            step = max(-DMAX, min(DMAX, bins))
            out.append(tok[f"d{step}"])
            bins -= step
            if bins == 0:
                return
            stats["chained"] += 1

    lengths = []
    qerr_max = 0.0
    chained = Counter()
    rows = []
    n_points = []
    for key, r in by_feat.items():
        # compose: every glyph's subpaths at its (dx,dy), origin-relative
        seq = [tok["<bos>"]]
        px, py = 0.0, 0.0  # origin-relative running point, in units
        stats = {"chained": 0}
        pts = 0
        for g in r["glyphs"]:
            for sub in glyph_paths.get(g["gid"], []):
                for cmd, cps in sub:
                    seq.append(tok[cmd])
                    for (x, y) in cps:
                        ax, ay = x + g["dx"], y + g["dy"]
                        bx = round((ax - px) / Q)
                        by = round((ay - py) / Q)
                        emit_delta(seq, bx, stats)
                        emit_delta(seq, by, stats)
                        nx, ny = px + bx * Q, py + by * Q
                        qerr_max = max(qerr_max, abs(nx - ax), abs(ny - ay))
                        px, py = nx, ny
                        pts += 1
                seq.append(tok["Z"])
        seq.append(tok["<eos>"])
        lengths.append(len(seq))
        chained[stats["chained"]] += 1
        n_points.append(pts)
        rows.append({
            "letters": r["letters"],
            "prev2": r.get("prev2"), "prev": r.get("prev"),
            "next": r.get("next"), "next2": r.get("next2"),
            "ddx": r.get("ddx"), "ddy": r.get("ddy"),
            "ox": r.get("ox"), "oy": r.get("oy"),
            "tokens": seq,
        })

    lengths.sort()
    n_points.sort()
    n = len(lengths)
    def pct(v, p):
        return v[min(n - 1, int(n * p))]
    total_tokens = sum(lengths)

    os.makedirs(OUT, exist_ok=True)
    json.dump(vocab, open(f"{OUT}/vocab.json", "w"))
    with open(f"{OUT}/sequences.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "q_units": Q,
        "delta_max_bins": DMAX,
        "vocab_size": len(vocab),
        "contexts": n,
        "tokens_total": total_tokens,
        "len_mean": total_tokens / n,
        "len_p50": pct(lengths, 0.50),
        "len_p95": pct(lengths, 0.95),
        "len_max": lengths[-1],
        "points_p50": pct(n_points, 0.50),
        "points_max": n_points[-1],
        "quant_err_max_units": qerr_max,
        "chained_delta_contexts": sum(v for k, v in chained.items() if k > 0),
    }
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)

    print(f"q = {Q} units/bin, delta range ±{DMAX} bins (±{Q*DMAX:.0f} units)")
    print(f"vocab: {len(vocab)} tokens")
    print(f"contexts: {n}")
    print(f"sequence length: mean {total_tokens/n:.0f}, p50 {pct(lengths,0.5)}, "
          f"p95 {pct(lengths,0.95)}, max {lengths[-1]}")
    print(f"outline points: p50 {pct(n_points,0.5)}, max {n_points[-1]}")
    print(f"dataset: {total_tokens:,} tokens total")
    print(f"max quantization error: {qerr_max:.2f} units "
          f"({qerr_max/1000*100:.2f}% of em)")
    print(f"contexts needing chained deltas: "
          f"{meta['chained_delta_contexts']} of {n}")


if __name__ == "__main__":
    main()
