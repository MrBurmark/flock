/**
AUTHOR: Moses Lee, Jason Burmark, Rachel Beasley
COMPILE: nvcc cuflock.cu utils.c -o cuflock -O3 -lm -arch=compute_20 -code=sm_20,sm_30,sm_35 
NOTE: compile with -fmad=false for degub/checking against serial
RUN: ./cuflock [input file]
DESCRIPTION: Runs a simple flocking simulation on the GPU
**/

#include "flock.h"
#include "utils.c"
#include "cudaUtils.cu"

__global__ void cuUpdateFlock(float *, int);
__global__ void cuApplyNeighborForce(float *, int);
__global__ void cuUpdateApplyNeighbor(float *, float *, int);

int main(int argc, char** argv)
{
	int i, nPoints;
	int ok;
    float *h_boids, *d_boidsB, *g_boids;
#if DOUBLEBUFFER
    float *d_boidsA, *tmpPtr;
#endif
    struct timeval tv;
    double t0, t1, t2;
#if !DOUBLEBUFFER
    double updateFlockTime = 0.0;
#endif
	double applyNeighborForceTime = 0.0;
    FILE *fp;

    if (argc != 2) {
        printf("Usage: ./cuflock [input file]\n");
        exit(1);
    }

	// read input file of initial conditions
	fp = fopen(argv[1], "r");
	ok = fscanf(fp, "%d", &nPoints);
	if (ok != 1) printf("Uh-oh\n");
	printf("Cuda - %d points, %i threads\n", nPoints, NUM_THREADS);
	h_boids = (float *) calloc(nPoints*6, sizeof(float));

	loadBoids(fp, h_boids, nPoints);

	g_boids = (float *) calloc(nPoints*6, sizeof(float));
	memcpy(g_boids, h_boids, (nPoints * 6) * sizeof(float));

    // allocate device memory
    cudaMalloc((void**) &d_boidsB, nPoints*6 * sizeof(float));
#if DOUBLEBUFFER
    cudaMalloc((void**) &d_boidsA, nPoints*6 * sizeof(float));
#endif

    gettimeofday(&tv, NULL);
	t0 = tv.tv_sec*1e6 + tv.tv_usec;

    // copy host memory to device
    cudaMemcpy(d_boidsB, h_boids, nPoints*6 * sizeof(float), cudaMemcpyHostToDevice);

	gettimeofday(&tv, NULL);
	t2 = tv.tv_sec*1e6 + tv.tv_usec;

	for (i=0; i<NUMCYCLES; i++) {

#if !DOUBLEBUFFER
		cuUpdateFlock<<<ceil(nPoints / (double)NUM_THREADS), NUM_THREADS>>>(d_boidsB, nPoints);
		cudaDeviceSynchronize();

		gettimeofday(&tv, NULL);
		t1 = tv.tv_sec*1e6 + tv.tv_usec;
		updateFlockTime += t1 - t2;

		cuApplyNeighborForce<<<ceil(nPoints / (double)NUM_THREADS), NUM_THREADS>>>(d_boidsB, nPoints);
		cudaDeviceSynchronize();

		gettimeofday(&tv, NULL);
		t2 = tv.tv_sec*1e6 + tv.tv_usec;
		applyNeighborForceTime += t2 - t1;

#else
		cuUpdateApplyNeighbor<<<ceil(nPoints / (double)NUM_THREADS), NUM_THREADS>>>(d_boidsA, d_boidsB, nPoints);

		// swap buffers
		tmpPtr = d_boidsA;
		d_boidsA = d_boidsB;
		d_boidsB = tmpPtr;

		cudaDeviceSynchronize();
		
		gettimeofday(&tv, NULL);
		t1 = tv.tv_sec*1e6 + tv.tv_usec;
		applyNeighborForceTime += t1 - t2;
		t2 = t1;
#endif
	}

#if DOUBLEBUFFER
	tmpPtr = d_boidsA;
	d_boidsA = d_boidsB;
	d_boidsB = tmpPtr;
#endif

#if !DOUBLEBUFFER
	// copy host memory to device
    cudaMemcpy(h_boids, d_boidsB, nPoints*6 * sizeof(float), cudaMemcpyDeviceToHost);
#else
    cudaMemcpy(h_boids, d_boidsB, nPoints*4 * sizeof(float), cudaMemcpyDeviceToHost);
#endif

	gettimeofday(&tv, NULL);
	t2 = tv.tv_sec*1e6 + tv.tv_usec;

#if !DOUBLEBUFFER
	printf("updateFlock: %f applyNeighborForce %f total %f\n",
		updateFlockTime, applyNeighborForceTime, t2-t0);
#else
	printf("updateFlock and applyNeighborForce %f total %f\n",
		applyNeighborForceTime, t2-t0);
#endif

#if DUMP
	// dump positions of points
	dumpBoids(h_boids, nPoints);
#endif

#if CHECK
	computeGold(g_boids, nPoints);
	printDiff(h_boids, g_boids, nPoints, .01);
#endif

    // clean up memory
    free(h_boids);
	free(g_boids);
    cudaFree(d_boidsB);
#if DOUBLEBUFFER
    cudaFree(d_boidsA);
#endif

    cudaThreadExit();
}
