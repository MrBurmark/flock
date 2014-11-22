/*
 * Filename: cudaUtils.cu
 * This file is for the GPU version of the flock simulations
 */

#include "flock.h"
// void updateFlock(float * b, int NP)
// compute new positions and velocities of flocking points
// b (input/output): array of positions/velocities/accelerations
// NP (input): number of points
//
// reads velocities and accelerations
// writes positions and velocities 

__global__ void cuUpdateFlock(float * b, int NP) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= NP) return;

	pos(b, i, 0, NP) += vel(b, i, 0, NP);
	pos(b, i, 1, NP) += vel(b, i, 1, NP);

	vel(b, i, 0, NP) += acc(b, i, 0, NP);
	vel(b, i, 1, NP) += acc(b, i, 1, NP);
}

// void applyNeighborForce(float *b, int NP)
//
// compute new accelerations of flocking points based on centroid
//        of neighborhood around each point
// b (input/output): array of positions/velocities/accelerations
// NP (input): number of points
//
// reads positions and velocities 
// writes accelerations

__global__ void cuApplyNeighborForce(float *b, int NP)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	float sumX = 0.0f, sumY = 0.0f;
	int count = 0;
	float sqX, sqY;
	float posx, posy, jposx, jposy;
	float diff;
	const float neighborDist = 50.0f;
	float amp;
	const float maxVel = 4.0f, maxForce = 0.03f;
	float f;

	if (i >= NP) return;

	posx = pos(b, i, 0, NP);
	posy = pos(b, i, 1, NP);

	for (j=0; j<NP; j++) {

		if (i==j) continue;

		jposx = pos(b, j, 0, NP);
		jposy = pos(b, j, 1, NP);

		sqX = posx - jposx;
		sqX *= sqX;
		sqY = posy - jposy;
		sqY *= sqY;

		diff = sqX + sqY;
		if (diff > neighborDist*neighborDist) continue;
		sumX += jposx;
		sumY += jposy;
		count++;
	}

	if (count != 0) {

		sumX /= count;
		sumY /= count;

		// centroid of neighborhood is now sumX, sumY
		sumX -= posx;
		sumY -= posy;

		amp = sqrt(sumX * sumX + sumY * sumY);
		sumX *= maxVel / amp;
		sumY *= maxVel / amp;

		sumX -= vel(b, i, 0, NP);
		sumY -= vel(b, i, 1, NP);
		amp = sqrt(sumX * sumX + sumY * sumY);

		if (amp > maxForce) {
			f = maxForce / amp;
			sumX *= f;
			sumY *= f;
		}
	}

	acc(b, i, 0, NP) = sumX;
	acc(b, i, 1, NP) = sumY;
}

// write buffer a, read buffer b
// removes necessity for device synchronize between kernels
// puts updating of birds after finding neighbors
// removes necessity of storing accelerations
__global__ void cuUpdateApplyNeighbor(float *a, float * b, int NP) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	float sumX = 0.0f, sumY = 0.0f;
	int count = 0;
	float sqX, sqY;
	float posx, posy, velx, vely, jposx, jposy;
	float diff;
	const float neighborDistSQR = 50.0f * 50.0f;
	float amp;
	const float maxVel = 4.0f, maxForce = 0.03f;
	float f;

	if (i >= NP) return;

	posx = pos(b, i, 0, NP);
	posy = pos(b, i, 1, NP);

	for (j=0; j<NP; j++) {

		if (i==j) continue;

		jposx = pos(b, j, 0, NP);
		jposy = pos(b, j, 1, NP);

		sqX = posx - jposx;
		sqX *= sqX;
		sqY = posy - jposy;
		sqY *= sqY;

		diff = sqX + sqY;
		if (diff > neighborDistSQR) continue;
		sumX += jposx;
		sumY += jposy;
		count++;
	}

	velx = vel(b, i, 0, NP);
	vely = vel(b, i, 1, NP);

	if (count != 0) {

		sumX /= count;
		sumY /= count;

		// centroid of neighborhood is now sumX, sumY
		sumX -= posx;
		sumY -= posy;

		amp = sqrt(sumX * sumX + sumY * sumY);
		sumX *= maxVel / amp;
		sumY *= maxVel / amp;

		sumX -= velx;
		sumY -= vely;
		amp = sqrt(sumX * sumX + sumY * sumY);

		if (amp > maxForce) {
			f = maxForce / amp;
			sumX *= f;
			sumY *= f;
		}
	}

	posx += velx;
	posy += vely;

	pos(a, i, 0, NP) = posx;
	pos(a, i, 1, NP) = posy;	

	velx += sumX;
	vely += sumY;

	vel(a, i, 0, NP) = velx;
	vel(a, i, 1, NP) = vely;
}

#if WARP
// can only be compiled with -arch=compute_30 or higher
// comments for cuUpdateApplyNeighbor apply
// gives one boid to one warp and reduces results
__global__ void cuUpdateApplyNeighborWarpReduce(float *a, float * b, int NP) {

	int i = blockIdx.x * (NUM_THREADS / WARPSIZE) + threadIdx.x / WARPSIZE;
	int warpId = threadIdx.x % WARPSIZE;
	int j;
	float sumX = 0.0f, sumY = 0.0f;
	int count = 0;
	float sqX, sqY;
	float posx, posy, velx, vely, jposx, jposy;
	float diff;
	const float neighborDistSQR = 50.0f * 50.0f;
	float amp;
	const float maxVel = 4.0f, maxForce = 0.03f;
	float f;

	if (i >= NP) return;

	posx = pos(b, i, 0, NP);
	posy = pos(b, i, 1, NP);

	for (j=warpId; j<NP; j+=WARPSIZE) {

		if (i==j) continue;

		jposx = pos(b, j, 0, NP);
		jposy = pos(b, j, 1, NP);

		sqX = posx - jposx;
		sqX *= sqX;
		sqY = posy - jposy;
		sqY *= sqY;

		diff = sqX + sqY;
		if (diff > neighborDistSQR) continue;
		sumX += jposx;
		sumY += jposy;
		count++;
	}

	for (j=WARPSIZE/2;j>=1;j/=2)
		sumX += __shfl_xor(sumX, j, WARPSIZE);
	for (j=WARPSIZE/2;j>=1;j/=2)
		sumY += __shfl_xor(sumY, j, WARPSIZE);
	for (j=WARPSIZE/2;j>=1;j/=2)
		count += __shfl_xor(count, j, WARPSIZE);

	if (warpId != 0) return;

	velx = vel(b, i, 0, NP);
	vely = vel(b, i, 1, NP);

	if (count != 0) {

		sumX /= count;
		sumY /= count;

		// centroid of neighborhood is now sumX, sumY
		sumX -= posx;
		sumY -= posy;

		amp = sqrt(sumX * sumX + sumY * sumY);
		sumX *= maxVel / amp;
		sumY *= maxVel / amp;

		sumX -= velx;
		sumY -= vely;
		amp = sqrt(sumX * sumX + sumY * sumY);

		if (amp > maxForce) {
			f = maxForce / amp;
			sumX *= f;
			sumY *= f;
		}
	}

	posx += velx;
	posy += vely;

	pos(a, i, 0, NP) = posx;
	pos(a, i, 1, NP) = posy;	

	velx += sumX;
	vely += sumY;

	vel(a, i, 0, NP) = velx;
	vel(a, i, 1, NP) = vely;
}
#endif