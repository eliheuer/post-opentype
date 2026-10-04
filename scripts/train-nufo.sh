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
#   NTF_LICENSE and NTF_NOTICE (the license and notice in the font's header),
#   FEATURES (cargo features for the trainer, such as cuda or metal)
set -e
src="${1:?usage: train-nufo.sh <source.nufo> <name> [epochs]}"
name="${2:?usage: train-nufo.sh <source.nufo> <name> [epochs]}"
epochs="${3:-400}"
export NTF_LR="${NTF_LR:-1e-3}"
export NTF_HAND_OS="${NTF_HAND_OS:-64}"
cd "$(dirname "$0")/.."

n=1
while [ -e "models/$name/$(printf %03d "$n")" ]; do n=$((n + 1)); done
version="$(printf %03d "$n")"
dir="models/$name/$version"
mkdir -p "$dir/base" "$dir/train"
cp -R "$src" "$dir/source.nufo"

cargo build --release -p neuraltype-distill
cargo build --release -p neuraltype-train ${FEATURES:+--features "$FEATURES"}

# The canvas every shape is drawn on. With no rows, this base adds
# nothing to the dataset but the canvas size.
cat > "$dir/base/fields-meta.json" <<META
{ "em_px": 64, "w": 155, "h": 219, "origin_x": 55.36, "origin_y": 108.48,
  "spread_px": 8.0, "upm": 1000.0, "shapes": 0 }
META
: > "$dir/base/dataset.jsonl"
: > "$dir/base/fields.bin"

target/release/distill hand "$dir/base" "$dir/fields" "$dir/source.nufo" > "$dir/distill.log"
cat "$dir/distill.log"
rm -r "$dir/base"
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# To a file, not through a pipe: a failed step must stop the script.
target/release/ntf-train "$dir/fields" "$dir/train" "$epochs" > "$dir/train.log"
tail -n 3 "$dir/train.log"
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

NAME="$name" VERSION="$version" DIR="$dir" SRC="$src" EPOCHS="$epochs" STARTED="$started" \
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
    },
    "dataset": {
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
json.dump(manifest, open(f"{d}/manifest.json", "w"), ensure_ascii=False, indent=2)
print(f"wrote {d}/manifest.json")
PY
echo "model $name $version: $dir/font.ntf"
