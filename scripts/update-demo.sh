#!/bin/sh
# Export the live checkpoint and push it to the blog demo.
set -e
cargo run --release -p neuraltype-train --features metal --bin ntf-train -- \
    export data/train-gulzar-run2 data/fields-gulzar-64 nastaliq-gulzar build/gulzar-live.ntf
cp build/gulzar-live.ntf ~/GH/repos/elih.net/public/demos/neuraltype/gulzar.ntf
cd ~/GH/repos/elih.net
pnpm build > /dev/null
git add public/demos/neuraltype/gulzar.ntf
git commit -m "nastaliq demo: newer training checkpoint"
git push
echo "demo font updated"
