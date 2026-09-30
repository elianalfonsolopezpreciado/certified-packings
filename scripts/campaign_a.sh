#!/bin/sh
# wait for the n=100 schedule to finish, then run evo for the remaining targets seeded with GPU basins
export KMP_DUPLICATE_LIB_OK=TRUE
while ! grep -q '^\[$' results/logs/mainA_n100.log; do sleep 20; done
python -m search.schedule --ns 120 150 200 116 138 139 159 188 --method evo --chunk 300 --max-chunks 5 --stall 3 \
  --workers 8 --iters 10 --seed 502 --tag mainA \
  --seed-files "results/runs/gpuL_sum_radii_n{n}/top.json" "results/best/sum_radii_n{n}.json"
