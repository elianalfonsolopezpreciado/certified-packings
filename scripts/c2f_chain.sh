#!/bin/sh
# coarse-to-fine chain from the certified N=600 candidate (seeded in results/autocorr1/cand_N600.json)
export KMP_DUPLICATE_LIB_OK=TRUE
prev=600
for N in 1200 2400 4800 9600 19200 38400; do
  python -m search.ac_c2f upsample --src $prev --dst $N --B 8 --noise 0.003 --stages 20 --steps 800 --tau0 1e-4 --tau1 1e-6 --seed 7 --top 2 --slp-seconds 120
  prev=$N
done
