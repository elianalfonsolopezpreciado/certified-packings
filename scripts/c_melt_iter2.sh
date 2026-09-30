#!/bin/sh
# longer iterated hot re-melting at N=4800, after the first batch finishes
export KMP_DUPLICATE_LIB_OK=TRUE
while [ "$(grep -c 'best float at N' results/logs/c_melt_iter_4800.log)" -lt 8 ]; do sleep 30; done
N=4800; seed=500
for round in 1 2 3 4; do
for combo in "1e-2 0.05" "3e-2 0.05" "1e-2 0.03" "2e-2 0.08"; do
  set -- $combo
  seed=$((seed+1))
  python -m search.ac_c2f upsample --src $N --dst $N --B 24 --noise $2 --stages 40 --steps 1000 --tau0 $1 --lr 0.01 --lr-end 0.001 --seed $seed --top 2 --slp-seconds 0
done
done
