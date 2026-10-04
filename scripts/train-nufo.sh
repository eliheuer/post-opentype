#!/bin/sh
# Train a neural font from a .nufo source alone: no teacher font, no
# starting weights. Each call makes a new numbered version under
# models/<name>/ and never changes an older one.
#
#   scripts/train-nufo.sh <source.nufo> <name> [epochs]
#
# A version holds everything needed to make it again:
#   manifest.json   what went in, the settings, the result
#   source.nufo/    the source as it was, outlines and labels
#   fields/         the training rows made from it
#   train/          the checkpoint and vocabulary
#   train.log       one line per epoch
#   font.ntf        the exported font
#
# Settings come from the environment and are recorded in the manifest:
#   NTF_LR (1e-3), NTF_HAND_OS (64, passes over the rows per epoch),
#   NTF_SEED (1, the seed of the random starting weights; GPU only),
#   NTF_INK_W (0, extra loss weight on the cells in and near the ink; it
#   gets a stalled run moving but makes strokes heavy, see ba-basic 003),
#   FROM_VERSION (export again from the training of an older version of
#   the same name, without training; for a changed header or exporter),
#   NTF_LICENSE and NTF_NOTICE (the license and notice in the font's header),
#   EM_PX (64, pixels of field per em; the canvas size itself is
#   measured from the largest labeled letter of the source),
#   FEATURES (cargo features for the trainer, such as cuda or metal),
#   TRAIN_HOST (an ssh host to train on; it needs a built checkout of this
#   repo at TRAIN_REPO, by default GH/repos/post-opentype in its home)
set -e
src="${1:?usage: train-nufo.sh <source.nufo> <name> [epochs]}"
name="${2:?usage: train-nufo.sh <source.nufo> <name> [epochs]}"
epochs="${3:-400}"
export NTF_LR="${NTF_LR:-1e-3}"
export NTF_HAND_OS="${NTF_HAND_OS:-64}"
export NTF_SEED="${NTF_SEED:-1}"
export NTF_INK_W="${NTF_INK_W:-0}"
cd "$(dirname "$0")/.."

n=1
while [ -e "models/$name/$(printf %03d "$n")" ]; do n=$((n + 1)); done
version="$(printf %03d "$n")"
dir="models/$name/$version"
mkdir -p "$dir/base" "$dir/train"
cp -R "$src" "$dir/source.nufo"

if [ -n "$FROM_VERSION" ]; then
    # No training: the rows, checkpoint, and log of the older version.
    from="models/$name/$FROM_VERSION"
    rm -r "$dir/source.nufo" "$dir/base" "$dir/train"
    cp -R "$from/source.nufo" "$from/fields" "$from/train" "$from/train.log" "$dir/"
    started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    cargo build --release -p neuraltype-train
else
    cargo build --release -p neuraltype-distill
    cargo build --release -p neuraltype-train ${FEATURES:+--features "$FEATURES"}

    # An empty base: no rows, and a canvas sized from the source. The
    # largest labeled letter sets the size, so new drawings need no
    # new setting. EM_PX is the detail: pixels of field per em.
    cat > "$dir/base/fields-meta.json" <<META
{ "em_px": ${EM_PX:-64}, "w": 0, "h": 0, "origin_x": 0, "origin_y": 0,
  "spread_px": 8.0, "upm": 1000.0, "shapes": 0, "auto_canvas": true }
META
    : > "$dir/base/dataset.jsonl"
    : > "$dir/base/fields.bin"

    target/release/distill hand "$dir/base" "$dir/fields" "$dir/source.nufo" > "$dir/distill.log"
    cat "$dir/distill.log"
    rm -r "$dir/base"
    started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    # To a file, not through a pipe: a failed step must stop the script.
    if [ -n "$TRAIN_HOST" ]; then
        # Train on another machine: send the rows, run its trainer, and bring
        # the checkpoint and log back. Everything else happens here.
        remote="${TRAIN_REPO:-GH/repos/post-opentype}"
        ssh "$TRAIN_HOST" "mkdir -p '$remote/$dir'"
        rsync -a "$dir/fields" "$dir/train" "$TRAIN_HOST:$remote/$dir/"
        # Every NTF_ setting goes along, so the remote run is the recorded one.
        settings="$(env | grep '^NTF_' | grep -v '^NTF_LICENSE=\|^NTF_NOTICE=' | sed "s/=\(.*\)/='\1'/" | tr '\n' ' ')"
        ssh "$TRAIN_HOST" "cd '$remote' && $settings \
            target/release/ntf-train '$dir/fields' '$dir/train' '$epochs' > '$dir/train.log'"
        rsync -a "$TRAIN_HOST:$remote/$dir/train" "$TRAIN_HOST:$remote/$dir/train.log" "$dir/"
    else
        target/release/ntf-train "$dir/fields" "$dir/train" "$epochs" > "$dir/train.log"
    fi
    tail -n 3 "$dir/train.log"
fi

# The font's header says where it came from. The exporter's default
# notice is the distilled font's, which is false here.
NTF_LICENSE="${NTF_LICENSE:-Not specified}" \
NTF_NOTICE="${NTF_NOTICE:-Trained from the source $name alone. No teacher font.}" \
python3 - "$dir/fields/header-extra.json" <<'PY'
import json, os, sys
path = sys.argv[1]
extra = json.load(open(path))
extra["license"] = os.environ["NTF_LICENSE"]
extra["notice"] = os.environ["NTF_NOTICE"]
json.dump(extra, open(path, "w"), ensure_ascii=False, indent=2)
PY
target/release/ntf-train export "$dir/train" "$dir/fields" "$name-$version" "$dir/font.ntf"

FROM_VERSION="$FROM_VERSION" NAME="$name" VERSION="$version" DIR="$dir" SRC="$src" EPOCHS="$epochs" STARTED="$started" \
python3 - <<'PY'
import hashlib, json, os, subprocess
d = os.environ["DIR"]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
def tree_sha(root):
    h = hashlib.sha256()
    for base, dirs, files in sorted(os.walk(root)):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(base, name)
            h.update(os.path.relpath(path, root).encode())
            h.update(sha(path).encode())
    return h.hexdigest()
log = open(f"{d}/train.log").read().splitlines()
epochs = [line for line in log if line.startswith("epoch")]
rows = [json.loads(line) for line in open(f"{d}/fields/dataset.jsonl")]
manifest = {
    "name": os.environ["NAME"],
    "version": os.environ["VERSION"],
    "started": os.environ["STARTED"],
    "teacher": None,
    "starting_weights": None,
    "training_reused_from": os.environ.get("FROM_VERSION") or None,
    "source": {
        "path": os.environ["SRC"],
        "snapshot": "source.nufo",
        "sha256": tree_sha(f"{d}/source.nufo"),
    },
    "code": {
        "commit": git("rev-parse", "HEAD"),
        "uncommitted_changes": git("status", "--porcelain").splitlines(),
    },
    "settings": {
        "epochs": int(os.environ["EPOCHS"]),
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("NTF_")},
        "device": next((l.split(": ", 1)[1] for l in log if l.startswith("device")), None),
        "train_host": os.environ.get("TRAIN_HOST") or None,
        "seed": next((l.split(": ", 1)[1] for l in log if l.startswith("seed")), None),
    },
    "dataset": {
        "canvas": json.load(open(f"{d}/fields/fields-meta.json")),
        "rows": len(rows),
        "clusters": [
            {k: r.get(k) for k in ("prev2", "prev", "letters", "next", "next2")} for r in rows
        ],
        "sha256": {f: sha(f"{d}/fields/{f}") for f in ("dataset.jsonl", "fields.bin")},
    },
    "result": {
        "last_epoch": epochs[-1] if epochs else None,
        "sha256": {
            "font.ntf": sha(f"{d}/font.ntf"),
            "checkpoint.safetensors": sha(f"{d}/train/checkpoint.safetensors"),
        },
    },
}
if manifest["training_reused_from"]:
    # The settings and code that made the weights are the older version's.
    older = json.load(open(f"{d}/../{manifest['training_reused_from']}/manifest.json"))
    manifest["settings"] = older["settings"]
    manifest["training_code"] = older["code"]
json.dump(manifest, open(f"{d}/manifest.json", "w"), ensure_ascii=False, indent=2)
print(f"wrote {d}/manifest.json")
PY
echo "model $name $version: $dir/font.ntf"
