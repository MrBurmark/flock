#!/bin/bash

for nt in 32 64 128 256 512 1024; do
	for ws in 1 2 4 8 16 32; do
		nvcc -D NUM_THREADS=$nt -D WARPSIZE=$ws -D CHECK=0 cuflock.cu utils.c -o cuflock -O3 -lm -arch=compute_30 -code=sm_30
		./cuflock in1024 >> test.log
	done
done
