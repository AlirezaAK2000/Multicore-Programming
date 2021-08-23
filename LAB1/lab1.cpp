
#include <stdio.h>
#include <math.h>
#include <omp.h>

const long int VERYBIG = 100000;
// ***********************************************************************
int main(void)
{

	// checks if openMP is available
	#ifndef _OPENMP
		printf("OpenMP is not supported, sorry!\n");
		getchar();
		return 0;
	#endif 


	int i;
	long int j, k, sum;
	double sumx, sumy, total;
	double starttime, elapsedtime;
	// -----------------------------------------------------------------------
	// Output a start message
	printf("parallel Timings for %d iterations\n\n", VERYBIG);
	// repeat experiment several times
	for (i = 0; i < 10; i++)
	{
		// get starting time56 x CHAPTER 3 PARALLEL STUDIO XE FOR THE IMPATIENT
		starttime = omp_get_wtime();
		// reset check sum & running total
		sum = 0;
		total = 0.0;
		// Work Loop, do some work by looping VERYBIG times
		//#pragma omp parallel for private(k, sumx, sumy) reduction(+:sum, total) num_threads(32)
		#pragma omp parallel for private(k , sumx , sumy)
		for (j = 0; j < VERYBIG; j++)
		{
			int num_threads = omp_get_num_threads();
			//printf("num threads : %d\n", num_threads);


			// increment check sum
			//printf("thread id : %d \n", omp_get_thread_num());
			#pragma omp critical
				sum += 1;
			// Calculate first arithmetic series
			sumx = 0.0;
			for (k = 0; k < j; k++)
				sumx = sumx + (double)k;
			// Calculate second arithmetic series
			sumy = 0.0;
			for (k = j; k > 0; k--)
				sumy = sumy + (double)k;
			if (sumx > 0.0) {
				#pragma omp critical
					total = total + 1.0 / sqrt(sumx);
			}

			if (sumy > 0.0) {
				#pragma omp critical
				total = total + 1.0 / sqrt(sumy);
			}
		}
		// get ending time and use it to determine elapsed time
		elapsedtime = omp_get_wtime() - starttime;
		// report elapsed time
		printf("Time Elapsed: %f Secs, Total = %lf, Check Sum = %ld , iteration : %d\n",
			elapsedtime, total, sum , i
		);
	}
	// return integer as required by function header
	getchar();
	return 0;
}
