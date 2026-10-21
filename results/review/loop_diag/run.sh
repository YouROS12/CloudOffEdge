#!/bin/bash
# R10 diagnosis: does the replay-only control degrade because of the fine-tuning recipe?
cd "$(dirname "$0")/../../.."
O=results/review/loop_diag
for cfg in "uniform:" "balanced:" "uniform:--freeze-bn" "balanced:--freeze-bn"; do
  s=${cfg%%:*}; f=${cfg#*:}; tag=${s}${f:+_frozenbn}
  python -u experiments/scripts/train_edge_distill.py loop --out $O --out-name control_${tag}.json \
     --variants replay --sampler $s $f > $O/control_${tag}.log 2>&1
done
echo done > $O/DONE
