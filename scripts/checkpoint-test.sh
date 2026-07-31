#!/bin/sh
# Export the live training checkpoint and render a model-vs-Gulzar
# comparison sheet. Run any time during training; safe to repeat.
#   scripts/checkpoint-test.sh [run-dir] [words...]
set -e
RUN=${1:-data/train-gulzar-run2}
shift 2>/dev/null || true
WORDS=${@:-"بسم الله الرحمن الرحيم"}
cargo run --release -p neuraltype-train --features metal --bin ntf-train -- \
    export "$RUN" data/fields-gulzar-64 nastaliq-gulzar build/gulzar-live.ntf
cargo run --release -p neuraltype-distill --bin distill -- \
    compare data/Gulzar-Regular.ttf build/gulzar-live.ntf build/checkpoint.pgm $WORDS
magick build/checkpoint.pgm -resize 250% build/checkpoint.png
echo "open build/checkpoint.png  (top: Gulzar, bottom: model)"
grep epoch data/train-gulzar-run2.log | tail -3
