/***********************

    flock.c

    Description: simple flocking simulation
    Compile: gcc flock.c -O3 -o flock -lm
    Use: ./flock [input file]

***********************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define NUMCYCLES 500

/* macros for accessing array of 2-D positions, velocities and accelerations */
/* DATA is particle array */
/* POINTINDEX is particle id */
/* NP is number of particles */
/* x is DIMINDEX=0, y is DIMINDEX=1 */

#define pos(DATA, POINTINDEX, DIMINDEX, NP) DATA[(DIMINDEX)*(NP) + POINTINDEX]
#define vel(DATA, POINTINDEX, DIMINDEX, NP) DATA[(DIMINDEX+2)*(NP) + POINTINDEX]
#define acc(DATA, POINTINDEX, DIMINDEX, NP) DATA[(DIMINDEX+4)*(NP) + POINTINDEX]

void updateFlock(float *, int);
void applyNeighborForce(float *, int);
void loadBoids(FILE *, float *, int);
void dumpBoids(float *, int);
void dumpAccs(float *, int);
int nPoints; // number of flocking "birds"

int main(int argc, char **argv) {
  float *boids; // array of 2D positions, velocities, accelerations of points
  int i;

  if (argc != 2) {
    printf("Usage: ./flock [input file]\n");
    return(0);
  }

  // read input file of initial conditions
  FILE *fp = fopen(argv[1], "r");
  fscanf(fp, "%d", &nPoints);
  printf("%d points\n", nPoints);
  boids = (float *) calloc(nPoints*6, sizeof(float));
  loadBoids(fp, boids, nPoints);

  for (i=0; i<nPoints; i++) {
    vel(boids, i, 0, nPoints) = 0.;
    vel(boids, i, 1, nPoints) = 0.;
    acc(boids, i, 0, nPoints) = 0.;
    acc(boids, i, 1, nPoints) = 0.;
  }


  // set up timer
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double t0 = tv.tv_sec*1e6 + tv.tv_usec;
  double t1, t2;

  t2 = tv.tv_sec*1e6 + tv.tv_usec;
  double updateFlockTime = 0.;
  double applyNeighborForceTime = 0.;

  for (i=0; i<NUMCYCLES; i++) {
    updateFlock(boids, nPoints);
    gettimeofday(&tv, NULL);
    t1 = tv.tv_sec*1e6 + tv.tv_usec;
    updateFlockTime += t1 - t2;
    applyNeighborForce(boids, nPoints);
    gettimeofday(&tv, NULL);
    t2 = tv.tv_sec*1e6 + tv.tv_usec;
    applyNeighborForceTime += t2 - t1;
  }

  printf("updateFlock: %f applyNeighborForce %f total %f\n",
	 updateFlockTime, applyNeighborForceTime, t2-t0);

  // dump positions of points
  dumpBoids(boids, nPoints);
}

// void updateFlock(float * b, int NP)
//
// compute new positions and velocities of flocking points
// b (input/output): array of positions/velocities/accelerations
// NP (input): number of points
//
// reads velocities and accelerations
// writes positions and velocities 

void updateFlock(float * b, int NP) {
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

void applyNeighborForce(float *b, int NP) {

  int i, j;
  for (i=0; i<NP; i++) {
    acc(b, i, 0, NP) = 0.;
    acc(b, i, 1, NP) = 0.;
    int count = 0;
    float sumX = 0., sumY = 0.;
    for (j=0; j<NP; j++) {
      if (i==j) continue;
      float sqX = pos(b, i, 0, NP) - pos(b, j, 0, NP);
      sqX *= sqX;
      float sqY = pos(b, i, 1, NP) - pos(b, j, 1, NP);
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

void loadBoids(FILE *fp, float *b, int NP) {
  int i, dummy, temp;
  float tx, ty;

  for (i=0; i<NP; i++) {
    temp = fscanf(fp, "%d%f%f", &dummy, &tx, &ty);
    if (temp != 3) {
      printf("Error! Input file format incorrect\n");
      return;
    }
    pos(b, i, 0, NP) = tx;
    pos(b, i, 1, NP) = ty;
  }
}

void dumpBoids(float *b, int NP) {

  FILE *fp = fopen("dump.out", "w");
  float x, y;
  int i;
  for (i=0; i<NP; i++) {
    x = pos(b, i, 0, NP);
    y = pos(b, i, 1, NP);
    fprintf(fp, "%d %f %f\n", i, x, y);
  }
  fclose(fp);
}

void dumpAccs(float *b, int NP) {

  FILE *fp = fopen("dumpAcc.out", "w");
  float x, y;
  int i;
  for (i=0; i<NP; i++) {
    x = acc(b, i, 0, NP);
    y = acc(b, i, 1, NP);
    fprintf(fp, "%d %f %f\n", i, x, y);
  }
  fclose(fp);
}

