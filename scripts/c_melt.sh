#!/bin/sh
export KMP_DUPLICATE_LIB_OK=TRUE
python -m search.ac_c2f upsample --src 600 --dst 1200 --B 32 --noise 0.05 --stages 40 --steps 1000 --tau0 3e-3 --lr 0.01 --lr-end 0.001 --seed 201 --top 3 --slp-seconds 100
python -m search.ac_c2f upsample --src 600 --dst 1200 --B 32 --noise 0.10 --stages 40 --steps 1000 --tau0 1e-2 --lr 0.01 --lr-end 0.001 --seed 202 --top 3 --slp-seconds 100
python -m search.ac_c2f upsample --src 300 --dst 600 --B 32 --noise 0.10 --stages 40 --steps 1000 --tau0 1e-2 --lr 0.01 --lr-end 0.001 --seed 203 --top 3 --slp-seconds 100
