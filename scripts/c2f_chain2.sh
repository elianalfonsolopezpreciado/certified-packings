#!/bin/sh
# continuation of the coarse-to-fine chain (no LP polish above N=1200: dense LP is impractical there)
export KMP_DUPLICATE_LIB_OK=TRUE
prev=2400
for N in 4800 9600 19200 38400; do
  python -m search.ac_c2f upsample --src $prev --dst $N --B 8 --noise 0.003 --stages 20 --steps 800 --tau0 1e-4 --tau1 1e-6 --seed 7 --top 2 --slp-seconds 0
  prev=$N
done
