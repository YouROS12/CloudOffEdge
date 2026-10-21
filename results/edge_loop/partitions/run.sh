#!/bin/bash
# Loop pilot over 5 random week-1/week-2 partitions; fp32 evaluation; original training's balanced sampler.
cd "$(dirname "$0")/../../.."
O=results/edge_loop/partitions
for s in 1 2 3 4 5; do
  python -u experiments/scripts/train_edge_distill.py loop --out $O --out-name split${s}.json --split-seed $s \
     --sampler balanced > $O/split${s}.log 2>&1
done
echo done > $O/DONE
