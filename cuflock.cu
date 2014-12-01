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
__global__ void cuUpdateApplyNeighborWarpReduceShared(float *, float *, int);
__global__ void cuUpdateApplyNeighborWarpReduce(float *, float *, int);

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
	if (ok != 1) printf("Couldn't read nPoints\n");

#if MANMEM
	printf("Managed Memory - ");
#endif
#if DOUBLEBUFFER
	printf("Double Buffered - ");
#if SHARED
	printf("Shared Warp Reduction %i - ", WARPSIZE);
#elif WARP
	printf("Warp Reduction %i - ", WARPSIZE);
#endif
#endif
	printf("Num Cycles %i - ", NUMCYCLES);
	printf("Cuda - %d points, %i threads\n", nPoints, NUM_THREADS);

#if !MANMEM
	h_boids = (float *) calloc(nPoints*6, sizeof(float));

	loadBoids(fp, h_boids, nPoints);

    // allocate device memory
    cudaMalloc((void**) &d_boidsB, nPoints*6 * sizeof(float));
#if DOUBLEBUFFER
    cudaMalloc((void**) &d_boidsA, nPoints*6 * sizeof(float));
#endif
#else
    // allocate managed device memory
    cudaMallocManaged((void**) &d_boidsB, nPoints*6 * sizeof(float));
    memset(d_boidsB, 0, nPoints*6 * sizeof(float));

    h_boids = d_boidsB;

    loadBoids(fp, d_boidsB, nPoints);

#if DOUBLEBUFFER
    cudaMallocManaged((void**) &d_boidsA, nPoints*6 * sizeof(float));
#endif
#endif

    g_boids = (float *) calloc(nPoints*6, sizeof(float));
	memcpy(g_boids, h_boids, (nPoints * 6) * sizeof(float));

    gettimeofday(&tv, NULL);
	t0 = tv.tv_sec*1e6 + tv.tv_usec;

#if !MANMEM
    // copy host memory to device
#if !DOUBLEBUFFER
    cudaMemcpy(d_boidsB, h_boids, nPoints*6 * sizeof(float), cudaMemcpyHostToDevice);
#else
    cudaMemcpy(d_boidsB, h_boids, nPoints*4 * sizeof(float), cudaMemcpyHostToDevice);
#endif
#endif

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
#if !WARP && !SHARED
		cuUpdateApplyNeighbor<<<ceil(nPoints / (double)NUM_THREADS), NUM_THREADS>>>(d_boidsA, d_boidsB, nPoints);
#elif SHARED
		cuUpdateApplyNeighborWarpReduceShared<<<ceil(nPoints / (double)(NUM_THREADS / WARPSIZE)), NUM_THREADS>>>(d_boidsA, d_boidsB, nPoints);
#elif WARP
		cuUpdateApplyNeighborWarpReduce<<<ceil(nPoints / (double)(NUM_THREADS / WARPSIZE)), NUM_THREADS>>>(d_boidsA, d_boidsB, nPoints);
#endif
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

#if !MANMEM
	// copy host memory to device
#if !DOUBLEBUFFER
    cudaMemcpy(h_boids, d_boidsB, nPoints*6 * sizeof(float), cudaMemcpyDeviceToHost);
#else
    cudaMemcpy(h_boids, d_boidsB, nPoints*4 * sizeof(float), cudaMemcpyDeviceToHost);
#endif
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
#else

#endif

#if CHECK
	computeGold(g_boids, nPoints);

	printDiff(h_boids, g_boids, nPoints, .01);
#endif

    // clean up memory
#if !MANMEM
    free(h_boids);
#endif
	free(g_boids);
    cudaFree(d_boidsB);
#if DOUBLEBUFFER
    cudaFree(d_boidsA);
#endif

    cudaThreadExit();
}
