#!/bin/sh
# second wave: after the first GPU sweep ends, GPU-large seeds for more n; CPU evo follows once the first campaign is done
export KMP_DUPLICATE_LIB_OK=TRUE
while [ ! -f results/runs/gpuL_sum_radii_n188/summary.json ]; do sleep 20; done
python -m search.gpu configs/gpu_large_targets2.yaml > results/logs/gpu_large_targets2.log 2>&1 &
while ! grep -q '^\[$' results/logs/mainA_rest.log; do sleep 20; done
python -m search.schedule --ns 110 130 140 160 170 180 250 --method evo --chunk 300 --max-chunks 5 --stall 3 --workers 8 --iters 10 --seed 503 --tag mainA \
  --seed-files "results/runs/gpuL_sum_radii_n{n}/top.json" "results/best/sum_radii_n{n}.json" > results/logs/mainA_wave2.log 2>&1
