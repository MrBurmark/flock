/**
AUTHOR: Moses Lee, Jason Burmark, Rachel Beasley

COMPILE: nvcc cuflock.cu utils.c cudaUtils.cu -o cuflock -O3 -lm -arch=compute_20 -code=sm_20,sm_30,sm_35 

**/

#include <stdio.h>
#include "utils.c"
#include "cudaUtils.cu"

int main(int argc, char** argv)
{
	int i;
	int ok;
    cudaEvent_t start, stop;
    float time;
    float *h_boids, *d_boids;
    struct timeval tv;
    double t0, t1, t2;
    double updateFlockTime = 0.0;
	double applyNeighborForceTime = 0.0;
    FILE *fp;

    if (argc != 2) {
        printf("Usage: ./cuflock [input file]\n");
        exit(1);
    }

	// read input file of initial conditions
	fp = fopen(argv[1], "r");
	ok = fscanf(fp, "%d", &nPoints);
	printf("Cuda - %d points, %i threads\n", nPoints, NUM_THREADS);
	h_boids = (float *) calloc(nPoints*6, sizeof(float));
	loadBoids(fp, h_boids, nPoints);

    // allocate device memory
    cudaMalloc((void**) &d_boids, nPoints*6 * sizeof(float));

    gettimeofday(&tv, NULL);
	t0 = tv.tv_sec*1e6 + tv.tv_usec;

    // copy host memory to device
    cudaMemcpy(d_boids, h_boids, nPoints*6 * sizeof(float), cudaMemcpyHostToDevice);
	
	gettimeofday(&tv, NULL);
	t2 = tv.tv_sec*1e6 + tv.tv_usec;

	for (i=0; i<NUMCYCLES; i++) {

		cuUpdateFlock<<<ceil(nPoints / (double)NUM_THREADS), NUM_THREADS>>>(d_boids, nPoints);
		cudaThreadSynchronize();

		gettimeofday(&tv, NULL);
		t1 = tv.tv_sec*1e6 + tv.tv_usec;
		updateFlockTime += t1 - t2;

		cuApplyNeighborForce<<<ceil(nPoints / (double)NUM_THREADS), NUM_THREADS>>>(d_boids, nPoints);
		cudaThreadSynchronize();

		gettimeofday(&tv, NULL);
		t2 = tv.tv_sec*1e6 + tv.tv_usec;
		applyNeighborForceTime += t2 - t1;
	}

	// copy host memory to device
    cudaMemcpy(h_boids, d_boids, nPoints*6 * sizeof(float), cudaMemcpyDeviceToHost);

	gettimeofday(&tv, NULL);
	t2 = tv.tv_sec*1e6 + tv.tv_usec;

	printf("updateFlock: %f applyNeighborForce %f total %f\n",
		updateFlockTime, applyNeighborForceTime, t2-t0);

	// dump positions of points
	dumpBoids(boids, nPoints);

    // clean up memory
    free(h_boids);
    cudaFree(d_boids);

    cudaThreadExit();
}
