/***********************

    flock.c

    Description: simple flocking simulation
    Compile: gcc flock.c -O3 -o flock -lm
    Use: ./flock [input file]

***********************/

#include "flock.h"

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

