#ifndef FLOCK_HOST_H
#define FLOCK_HOST_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define NUMCYCLES 500

#define NUM_THREADS 128

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

#endif
