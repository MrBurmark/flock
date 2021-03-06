#ifndef FLOCK_H
#define FLOCK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifndef NUMCYCLES
#define NUMCYCLES 500
#endif
#ifndef DUMP
#define DUMP 1
#endif
#ifndef CHECK
#define CHECK 0
#endif
#ifndef MANMEM
#define MANMEM 0
#endif
#ifndef DOUBLEBUFFER
#define DOUBLEBUFFER 0
#endif
#ifndef SHARED // takes precedence over WARP
#define SHARED 0 // requires DOUBLEBUFFER
#endif
#ifndef WARP // will not compile if architecture not 30 or higher with this option
#define WARP 0 // requires DOUBLEBUFFER
#endif
#ifndef WARPSIZE
#define WARPSIZE 16 // power of 2, <= 32, evenly divides NUM_THREADS
#endif
#ifndef NUM_THREADS
#define NUM_THREADS 128 // 128 good for most
#endif

// Look in test.txt for run configurations and times

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
