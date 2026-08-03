#!/bin/sh
# Overnight resolution jump: wait for the running fine-tune, publish
# its result to the local demo, regenerate fields at 96 px/em on the
# v2 corpus, and start the from-scratch 96 px training run.
set -e
cd "$(dirname "$0")/.."

while pgrep -f "ntf-train data" >/dev/null; do sleep 120; done
echo "=== polish run finished; exporting 64px final ==="
target/release/ntf-train export data/train-gulzar-run2 data/fields-gulzar-64 nastaliq-gulzar build/gulzar-live.ntf
cp build/gulzar-live.ntf "$HOME/GH/repos/elih.net/public/demos/neuraltype/gulzar.ntf"
cd "$HOME/GH/repos/elih.net" && pnpm build >/dev/null 2>&1 && cd -

echo "=== regenerating fields at 96 px/em ==="
target/release/distill fields data/extract-gulzar data/fields-gulzar-96 96

echo "=== starting 96 px training ==="
mkdir -p data/train-gulzar-96
NTF_OVERSAMPLE=8 target/release/ntf-train data/fields-gulzar-96 data/train-gulzar-96 60 2>&1 | tee data/train-gulzar-96.log | tail -3
