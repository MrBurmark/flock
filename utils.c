
#include "flock.h"

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

void printresults(float * fp, int num_elem){
	int i;
	for(i = 0; i < num_elem; i++){
			printf("These are the results: %f at %d\n", fp[i], i);

	}

}

//------------------------------------------------------------------
// For checking solutions
//------------------------------------------------------------------


void computeGold(float * boids, int nPoints){
	int i;

	for (i=0; i<nPoints; i++) {
		vel(boids, i, 0, nPoints) = 0.;
		vel(boids, i, 1, nPoints) = 0.;
		acc(boids, i, 0, nPoints) = 0.;
		acc(boids, i, 1, nPoints) = 0.;
	}

	for (i=0; i<NUMCYCLES; i++) {
		updateFlock(boids, nPoints);
		applyNeighborForce(boids, nPoints);
	}

}

void printDiff(float *data1, float *data2, int iListLength, float fListTol)
{
		float h_x;
		float h_y;
		float g_x;
		float g_y;
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,u;
    int error_count=0;
        u = 1;
        for (i = 0; i < iListLength; i++) 
        {
			h_x = pos(data1, i, 0, iListLength);
			h_y = pos(data1, i, 1, iListLength);
			g_x = pos(data2, i, 0, iListLength);
			g_y = pos(data2, i, 1, iListLength);

						
            float fDiff = fabs(h_x - g_x) / g_x;

            if (fDiff > fListTol || isnan(fDiff)) 
            {                
                if (error_count < iListLength)
                {
                    if (u)
                    {
                        printf("\n  Row %d:\n", j);
                    }
                    printf("    Locx(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, h_x, g_x, fDiff);
                    u = 0;
                }
                error_count++;
            }

			fDiff = fabs(h_y - g_y) / g_y;
			
			if (fDiff > fListTol || isnan(fDiff)) 
            {                
                if (error_count < iListLength)
                {
                    if (u)
                    {
                        printf("\n  Row %d:\n", j);
                    }
                    printf("    Locy(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, h_y, g_y, fDiff);
                    u = 0;
                }
                error_count++;
            }
        }
    printf(" \n  Total Errors = %d\n\n", error_count);
}

