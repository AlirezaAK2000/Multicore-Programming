// Let it be.
#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

void fill_array(int *a, size_t n);
void prefix_sum(int *a, size_t n);
void print_array(int *a, size_t n);
void s_prefix_sum(int *a, size_t n);
void p_prefix_sum(int *a, size_t n);
void h_prefix_sum(int *a, size_t n);

#define NUM_THREADS 8
#define EXP_NUM 5

int main(int argc, char *argv[])
{

#ifndef _OPENMP
	printf("OpenMP is not supported, sorry!\n");
	getchar();
	return 0;
#endif

	unsigned int p;

	printf("[-] enter 1 if you want parallelism otherwise enter 0 : \n");
	scanf("%uld\n", &p);

	unsigned int n = 0;
	printf("[-] Please enter N: ");
	scanf("%uld\n", &n);
	double summ = 0;

	for(int i = 0 ; i < EXP_NUM ; i++)
	{	
		int *a = (int *)malloc(n * sizeof a);

		fill_array(a, n);

		if (p == 1)
		{
			double start = omp_get_wtime();
			p_prefix_sum(a, n);
			double end = omp_get_wtime() - start;
			printf("parallel time for %d threads : %lf\n", NUM_THREADS, end);
			summ += end;
		}
		else if(p == 0)
		{
			double start = omp_get_wtime();
			s_prefix_sum(a, n);
			double end = omp_get_wtime() - start;
			printf("serial time : %lf\n", end);
			summ += end;

		}else if(p == 2){
			double start = omp_get_wtime();
			h_prefix_sum(a, n);
			double end = omp_get_wtime() - start;
			printf("parallel time for %d threads (h&s): %lf\n", NUM_THREADS, end);
			summ += end;
		}

		free(a);

	}
	printf("average tiem : %f \n" , (summ/EXP_NUM));

	return EXIT_SUCCESS;
}

void p_prefix_sum(int *a, size_t n)
{
	int end[NUM_THREADS];

#pragma omp parallel num_threads(NUM_THREADS)
	{
		int th_num = omp_get_thread_num();

		int inedx = -1;
#pragma omp for schedule(static)
		for (int i = 1; i < n; i++)
		{
			a[i] = a[i] + a[i - 1];
			inedx = i;
		}
		end[th_num] = inedx;

#pragma omp barrier

#pragma omp single
		{
			for (int k = 0; k < NUM_THREADS - 1; k++)
			{
				for (int j = k + 1; j < NUM_THREADS; j++)
				{
					a[end[j]] += a[end[k]];
				}
			}
		}

#pragma omp for
		for (int k = 0; k < NUM_THREADS - 1; k++)
		{
			int start = end[k] + 1;
			int endd = end[k + 1] - 1;
			int last_val = a[end[k]];
			for (int j = start; j < endd; j++)
				a[j] += last_val;
		}
	}
}
void s_prefix_sum(int *a, size_t n)
{
	for (int i = 1; i < n; ++i)
	{
		a[i] = a[i] + a[i - 1];
	}
}

void h_prefix_sum(int *a, size_t n)
{
	for(long step = 1 ; step < n ; step *= 2){
		int tmp[n];

		#pragma omp parallel num_threads(NUM_THREADS)
		{
			#pragma omp single
			{
				for(int i = 0 ; i < n - step ; i++){
					#pragma omp task
					{
						tmp[i + step] = a[i] + a[i + step];
					}
				}				
			}

			#pragma omp barrier

			#pragma omp for 
			for(int i = 0 ; i < n ; i++)
			{
				a[i] = tmp[i];
			}

		}
	}
}

void print_array(int *a, size_t n)
{
	int i;
	printf("[-] array: ");
	for (i = 0; i < n; ++i)
	{
		printf("%d, ", a[i]);
	}
	printf("\b\b  \n");
}

void fill_array(int *a, size_t n)
{
	int i;
	for (i = 0; i < n; ++i)
	{
		a[i] = i + 1;
	}
}
