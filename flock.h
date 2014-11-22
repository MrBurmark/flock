#ifndef FLOCK_H
#define FLOCK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifndef NUMCYCLES
#define NUMCYCLES 500
#endif

#ifndef DOUBLEBUFFER
#define DOUBLEBUFFER 1
#endif

#ifndef DUMP
#define DUMP 0
#endif

#ifndef CHECK
#define CHECK 1
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 128
#endif

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
void printresults(float *, int);
void computeGold(float *, int);
void printDiff(float *, float *, int, float);

#endif
