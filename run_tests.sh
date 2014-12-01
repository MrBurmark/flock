#!/bin/bash
for input in in1000 in1024 in1028; do
	for nt in 4 8 16 32 64 128 256; do
		for run in 1 2 3 4 5; do
			nvcc -D NUM_THREADS=$nt -D WARPSIZE=4 -D DOUBLEBUFFER=0 -D SHARED=0 -D WARP=0 -D MANMEM=0 -D DUMP=0 -D CHECK=0 cuflock.cu utils.c -o cuflock -fmad=false -O3 -lm -arch=compute_20 -code=sm_20,sm_30,sm_35
			./cuflock $input
		done
	done
done
