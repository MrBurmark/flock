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
  int i;
  for (i=0; i<NP; i++) {
    pos(b, i, 0, NP) += vel(b, i, 0, NP);
    pos(b, i, 1, NP) += vel(b, i, 1, NP);

    vel(b, i, 0, NP) += acc(b, i, 0, NP);
    vel(b, i, 1, NP) += acc(b, i, 1, NP);
  }
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

  int i, j;
  for (i=0; i<NP; i++) {
    acc(b, i, 0, NP) = 0.;
    acc(b, i, 1, NP) = 0.;
    int count = 0;
    float sumX = 0., sumY = 0.;
    for (j=0; j<NP; j++) {
      if (i==j) continue;
      float sqX = pos(b, i, 0, NP) - pos(b, j, 0, NP);
      sqY *= sqY;
      
      float diff = sqrt(sqX + sqY);
      float neighborDist = 50;
      if (diff > neighborDist) continue;
      sumX += pos(b, j, 0, NP);
      sumY += pos(b, j, 1, NP);
      count++;
    }
    if (count == 0) continue;
    sumX /= count;
    sumY /= count;

    // centroid of neighborhood is now sumX, sumY
    sumX -= pos(b, i, 0, NP);
    sumY -= pos(b, i, 1, NP);
    
    float amp = sqrt(sumX * sumX + sumY * sumY);
    float maxVel = 4;
    sumX *= maxVel / amp;
    sumY *= maxVel / amp;
    
    sumX -= vel(b, i, 0, NP);
    sumY -= vel(b, i, 1, NP);
    amp = sqrt(sumX * sumX + sumY * sumY);
    
    float maxForce = .03;
    if (amp > maxForce) {
      float f = maxForce / amp;
      sumX *= f;
      sumY *= f;
    }
    acc(b, i, 0, NP) += sumX;
    acc(b, i, 1, NP) += sumY;
  }
}
