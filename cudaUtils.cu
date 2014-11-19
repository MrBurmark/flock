#include "flock.h"

// void updateFlock(float * b, int NP)
//
// compute new positions and velocities of flocking points
// b (input/output): array of positions/velocities/accelerations
// NP (input): number of points
//
// reads velocities and accelerations
// writes positions and velocities 

__global__ void cuUpdateFlock(float * b, int NP) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

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

__global__ void cuApplyNeighborForce(float *b, int NP) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	float sumX = 0.0f, sumY = 0.0f;
	int count = 0;
	float sqX, sqY;
	float diff;
	const float neighborDist = 50.0f;
	float amp;
	const float maxVel = 4.0f, maxForce = 0.03f;
	float f;

	for (j=0; j<NP; j++) {
		if (i==j) continue;
		sqX = pos(b, i, 0, NP) - pos(b, j, 0, NP);
		sqX *= sqX;
		sqY = pos(b, i, 1, NP) - pos(b, j, 1, NP);
		sqY *= sqY;

		diff = sqX + sqY;
		if (diff > neighborDist*neighborDist) continue;
		sumX += pos(b, j, 0, NP);
		sumY += pos(b, j, 1, NP);
		count++;
	}

	if (count != 0) {

		sumX /= count;
		sumY /= count;

		// centroid of neighborhood is now sumX, sumY
		sumX -= pos(b, i, 0, NP);
		sumY -= pos(b, i, 1, NP);

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
