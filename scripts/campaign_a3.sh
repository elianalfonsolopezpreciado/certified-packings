#!/bin/sh
# push phase: keep-going evo on the only standing improvement (120) and the closest near-misses, different seed
export KMP_DUPLICATE_LIB_OK=TRUE
while ! grep -q '^\[$' results/logs/mainA_wave2.log; do sleep 20; done
python -m search.schedule --ns 120 110 130 138 116 --method evo --chunk 300 --max-chunks 5 --stall 3 --workers 10 --iters 10 --seed 504 --tag pushA --keep-going \
  --seed-files "results/runs/gpuL_sum_radii_n{n}/top.json" "results/best/sum_radii_n{n}.json" > results/logs/pushA.log 2>&1
