#!/bin/sh
# third wave: GPU seeding now; CPU evo after the push phase
export KMP_DUPLICATE_LIB_OK=TRUE
python -m search.gpu configs/gpu_large_targets3.yaml > results/logs/gpu_large_targets3.log 2>&1
while ! grep -q '^\[$' results/logs/pushA.log; do sleep 30; done
python -m search.schedule --ns 300 297 301 275 --method evo --chunk 300 --max-chunks 5 --stall 3 --workers 10 --iters 6 --seed 505 --tag mainA \
  --seed-files "results/runs/gpuL_sum_radii_n{n}/top.json" "results/best/sum_radii_n{n}.json" > results/logs/mainA_wave3.log 2>&1
