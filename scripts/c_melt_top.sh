#!/bin/sh
# continue hot re-melting from the best N=9600 solution up to N=38400
export KMP_DUPLICATE_LIB_OK=TRUE
seed=700
python -m search.ac_c2f upsample --src 9600 --dst 19200 --B 8 --noise 0.05 --stages 40 --steps 1000 --tau0 1e-2 --lr 0.01 --lr-end 0.001 --seed 701 --top 2 --slp-seconds 0
for combo in "1e-2 0.05" "3e-2 0.05" "1e-2 0.03" "2e-2 0.08" "1e-2 0.05" "1e-2 0.03"; do
  set -- $combo; seed=$((seed+1))
  python -m search.ac_c2f upsample --src 19200 --dst 19200 --B 8 --noise $2 --stages 40 --steps 1000 --tau0 $1 --lr 0.01 --lr-end 0.001 --seed $seed --top 2 --slp-seconds 0
done
python -m search.ac_c2f upsample --src 19200 --dst 38400 --B 4 --noise 0.05 --stages 40 --steps 1000 --tau0 1e-2 --lr 0.01 --lr-end 0.001 --seed 799 --top 2 --slp-seconds 0
