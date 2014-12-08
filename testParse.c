
#define DUMP "parse.out"
#define BASIC 0

#include <stdio.h>
#include <stdlib.h>

struct run {
double totalTime;
double updateTime;
double forceTime;
};

int main(int argc, char **argv) {

	int i=0, j=0, k=0;
	int nw=0, nc=0, np=0, nt=0;
	int nwold=-1, ncold = -1, npold = -1, ntold = -1;
	int ok = 1;
	double uf=0.0, anf=0.0, tt=0.0;
	struct run data[4][9][5];
	FILE *fp;

	if (argc != 2) exit(1);

	fp = fopen(argv[1], "r");

	while (ok != EOF) {

#if 0
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

#if !DOUBLEBUFFER
	printf("updateFlock: %f applyNeighborForce %f total %f\n",
		updateFlockTime, applyNeighborForceTime, t2-t0);
#else
	printf("updateFlock and applyNeighborForce %f total %f\n",
		applyNeighborForceTime, t2-t0);
#endif
#endif

#if BASIC
		ok = fscanf(fp, "Num Cycles %i - Cuda - %d points, %i threads\nupdateFlock: %lf applyNeighborForce %lf total %lf\n", &nc, &np, &nt, &uf, &anf, &tt);

		if (ok == 6) {
			if (np == 257) i = 0;
			else if (np == 1000) i = 1;
			else if (np == 1024) i = 2;
			else if (np == 1028) i = 3;
			else continue;

			if (nt == 4) j = 0;
			else if (nt == 8) j = 1;
			else if (nt == 16) j = 2;
			else if (nt == 32) j = 3;
			else if (nt == 64) j = 4;
			else if (nt == 128) j = 5;
			else if (nt == 256) j = 6;
			else if (nt == 512) j = 7;
			else if (nt == 1024) j = 8;
			else continue;

			if (k >= 5 || nc != ncold || np != npold || nt != ntold) {
				ncold = nc;
				npold = np;
				ntold = nt;
				k = 0;
			} else k++;

			data[i][j][k].totalTime = tt;
			data[i][j][k].updateTime = uf;
			data[i][j][k].forceTime = anf;
		}
#else

		ok = fscanf(fp, "Double Buffered - Shared Warp Reduction %i - Num Cycles %i - Cuda - %i points, %i threads\nupdateFlock and applyNeighborForce %lf total %lf\n", &nw, &nc, &np, &nt, &anf, &tt);

		if (ok == 6) {
			if (np == 257) i = 0;
			else if (np == 1000) i = 1;
			else if (np == 1024) i = 2;
			else if (np == 1028) i = 3;
			else continue;

			if (nt == 4) j = 0;
			else if (nt == 8) j = 1;
			else if (nt == 16) j = 2;
			else if (nt == 32) j = 3;
			else if (nt == 64) j = 4;
			else if (nt == 128) j = 5;
			else if (nt == 256) j = 6;
			else if (nt == 512) j = 7;
			else if (nt == 1024) j = 8;
			else continue;

			if (k >= 5 || nw != nwold || nc != ncold || np != npold || nt != ntold) {
				nwold = nw;
				ncold = nc;
				npold = np;
				ntold = nt;
				k = 0;
			} else k++;

			data[i][j][k].totalTime = tt;
			data[i][j][k].forceTime = anf;
		}
#endif
	}

	fclose(fp);

	fp = fopen(DUMP, "w");

#if BASIC
	fprintf(fp, "update flock times\n");
	for (i=0; i < 4; i++) {
		if (i == 0) np = 257;
		else if (i == 1) np = 1000;
		else if (i == 2) np = 1024;
		else if (i == 3) np = 1028;
		fprintf(fp, "input in%i\n", np);
		for (j=0; j< 9; j++) {
			if (j == 0) nt = 4;
			else if (j == 1) nt = 8;
			else if (j == 2) nt = 16;
			else if (j == 3) nt = 32;
			else if (j == 4) nt = 64;
			else if (j == 5) nt = 128;
			else if (j == 6) nt = 256;
			else if (j == 7) nt = 512;
			else if (j == 8) nt = 1024;
			// fprintf(fp, "num threads%i\n", nt);
			for (k=0; k<5; k++) {
				fprintf(fp, "%f\t", data[i][j][k].updateTime);
			}
			fprintf(fp, "\n");
		}
	}
	fprintf(fp, "\n");
#endif

	fprintf(fp, "apply neighbor force times\n");
	for (i=0; i < 4; i++) {
		if (i == 0) np = 257;
		else if (i == 1) np = 1000;
		else if (i == 2) np = 1024;
		else if (i == 3) np = 1028;
		fprintf(fp, "input in%i\n", np);
		for (j=0; j< 9; j++) {
			if (j == 0) nt = 4;
			else if (j == 1) nt = 8;
			else if (j == 2) nt = 16;
			else if (j == 3) nt = 32;
			else if (j == 4) nt = 64;
			else if (j == 5) nt = 128;
			else if (j == 6) nt = 256;
			else if (j == 7) nt = 512;
			else if (j == 8) nt = 1024;
			// fprintf(fp, "num threads%i\n", nt);
			for (k=0; k<5; k++) {
				fprintf(fp, "%f\t", data[i][j][k].forceTime);
			}
			fprintf(fp, "\n");
		}
	}
	fprintf(fp, "\n");

	fprintf(fp, "total times\n");
	for (i=0; i < 4; i++) {
		if (i == 0) np = 257;
		else if (i == 1) np = 1000;
		else if (i == 2) np = 1024;
		else if (i == 3) np = 1028;
		fprintf(fp, "input in%i\n", np);
		for (j=0; j< 9; j++) {
			if (j == 0) nt = 4;
			else if (j == 1) nt = 8;
			else if (j == 2) nt = 16;
			else if (j == 3) nt = 32;
			else if (j == 4) nt = 64;
			else if (j == 5) nt = 128;
			else if (j == 6) nt = 256;
			else if (j == 7) nt = 512;
			else if (j == 8) nt = 1024;
			// fprintf(fp, "num threads%i\n", nt);
			for (k=0; k<5; k++) {
				fprintf(fp, "%f\t", data[i][j][k].totalTime);
			}
			fprintf(fp, "\n");
		}
	}
	fprintf(fp, "\n");

	fclose(fp);
	
}