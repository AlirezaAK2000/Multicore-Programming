// Example Program
// Optimizes code for maximum speed
#pragma optimize( "2", on )
#include <stdio.h>
#include <windows.h>
#include <mmsystem.h>
#include <math.h>
#include <omp.h>
// Adds an additional library so that timeGetTime() can be used
#pragma comment(lib, "winmm.lib")
const long int VERYBIG = 100000;
// ***********************************************************************
int main(void)
{
	int i;
	long int j, k, sum;
	double sumx, sumy, total;
	DWORD starttime, elapsedtime;
	// -----------------------------------------------------------------------
	// Output a start message
	printf("None Parallel Timings for %d iterations\n\n", VERYBIG);
	// repeat experiment several times
	for (i = 0; i < 1; i++)
	{
		// get starting time56 x CHAPTER 3 PARALLEL STUDIO XE FOR THE IMPATIENT
		starttime = timeGetTime();
		// reset check sum & running total
		sum = 0;
		total = 0.0;
		// Work Loop, do some work by looping VERYBIG times
#pragma omp parallel for num_threads(8) private(sumx , sumy , k) reduction(+ : sum , total) schedule(dynamic , 2000)
		for (j = 0; j < VERYBIG; j++)
		{
			// increment check sum
			sum += 1;
			// Calculate first arithmetic series
			sumx = 0.0;
			for (k = 0; k < j; k++)
				sumx = sumx + (double)k;
			// Calculate second arithmetic series
			sumy = 0.0;
			for (k = j; k > 0; k--)
				sumy = sumy + (double)k;
			if (sumx > 0.0)total = total + 1.0 / sqrt(sumx);
			if (sumy > 0.0)total = total + 1.0 / sqrt(sumy);
		}
		// get ending time and use it to determine elapsed time
		elapsedtime = timeGetTime() - starttime;
		// report elapsed time
		printf("Time Elapsed % 10d mSecs Total = %lf Check Sum = %ld\n",
			(int)elapsedtime, total, sum);
	}
	// return integer as required by function header
	return 0;
}
// **********************************************************************