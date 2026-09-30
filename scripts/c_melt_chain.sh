#!/bin/sh
# partial re-melting up the ladder, GPU only (no dense-LP polish): tau restarts at 3e-3 with 5% noise at every doubling
export KMP_DUPLICATE_LIB_OK=TRUE
prev=1200
for spec in "2400 32" "4800 24" "9600 16" "19200 8" "38400 4"; do
  set -- $spec
  N=$1; B=$2
  python -m search.ac_c2f upsample --src $prev --dst $N --B $B --noise 0.05 --stages 40 --steps 1000 --tau0 3e-3 --lr 0.01 --lr-end 0.001 --seed 301 --top 2 --slp-seconds 0
  prev=$N
done
