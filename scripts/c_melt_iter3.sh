#!/bin/sh
# hot iterated re-melting at N=9600 (parallel to the N=4800 loop)
export KMP_DUPLICATE_LIB_OK=TRUE
N=9600; seed=600
for round in 1 2 3; do
for combo in "1e-2 0.05" "3e-2 0.05" "1e-2 0.03" "2e-2 0.08"; do
  set -- $combo
  seed=$((seed+1))
  python -m search.ac_c2f upsample --src $N --dst $N --B 12 --noise $2 --stages 40 --steps 1000 --tau0 $1 --lr 0.01 --lr-end 0.001 --seed $seed --top 2 --slp-seconds 0
done
done
