#!/bin/sh
# iterated partial re-melting at fixed N (src == dst): several (tau0, noise) combinations, best kept by save_cand
export KMP_DUPLICATE_LIB_OK=TRUE
N=${1:-4800}
seed=400
for round in 1 2; do
for combo in "1e-3 0.03" "3e-3 0.05" "1e-2 0.05" "3e-3 0.10"; do
  set -- $combo
  seed=$((seed+1))
  python -m search.ac_c2f upsample --src $N --dst $N --B 24 --noise $2 --stages 40 --steps 1000 --tau0 $1 --lr 0.01 --lr-end 0.001 --seed $seed --top 2 --slp-seconds 0
done
done
