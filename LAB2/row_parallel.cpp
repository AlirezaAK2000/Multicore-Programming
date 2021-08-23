/*
*	In His Exalted Name
*	Matrix Addition - Sequential Code
*	Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	15/04/2018
*/

// Let it be.
#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

typedef struct {
	int* A, * B, * C;
	int n, m;
} DataSet;

void fillDataSet(DataSet* dataSet);
void printDataSet(DataSet dataSet);
void closeDataSet(DataSet dataSet);
void add(DataSet dataSet);

#define EXP_NUM 10
#define NUM_THREADS 8

int main(int argc, char* argv[]) {
	#ifndef _OPENMP
		printf("OpenMP is not supported, sorry!\n");
		getchar();
		return 0;
	#endif 
	double elapsed_time_sum = 0.0;
	DataSet dataSet;
	if (argc < 3) {
		printf("[-] Invalid No. of arguments.\n");
		printf("[-] Try -> <n> <m> \n");
		printf(">>> ");
		scanf("%d %d", &dataSet.n, &dataSet.m);
	}
	else {
		dataSet.n = atoi(argv[1]);
		dataSet.m = atoi(argv[2]);
	}
	fillDataSet(&dataSet);
	for (int i = 0; i < EXP_NUM; i++) {
		double starttime = omp_get_wtime();
		add(dataSet);
		double elapsedtime = omp_get_wtime() - starttime;
		printf("Time Elapsed: %f Secs , for adding two (%d , %d ) dimensional matrix\n",
			elapsedtime, dataSet.n, dataSet.m
		);
		elapsed_time_sum += elapsedtime;
		//printDataSet(dataSet);
	}
	printf("average running time : %f\n", elapsed_time_sum / EXP_NUM);
	closeDataSet(dataSet);
	//system("PAUSE");
	return EXIT_SUCCESS;
}

void fillDataSet(DataSet* dataSet) {
	int i, j;

	dataSet->A = (int*)malloc(sizeof(int) * dataSet->n * dataSet->m);
	dataSet->B = (int*)malloc(sizeof(int) * dataSet->n * dataSet->m);
	dataSet->C = (int*)malloc(sizeof(int) * dataSet->n * dataSet->m);

	srand(time(NULL));

	for (i = 0; i < dataSet->n; i++) {
		for (j = 0; j < dataSet->m; j++) {
			dataSet->A[i * dataSet->m + j] = rand() % 100;
			dataSet->B[i * dataSet->m + j] = rand() % 100;
		}
	}
}

void printDataSet(DataSet dataSet) {
	int i, j;

	printf("[-] Matrix A\n");
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			printf("%-4d", dataSet.A[i * dataSet.m + j]);
		}
		putchar('\n');
	}

	printf("[-] Matrix B\n");
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			printf("%-4d", dataSet.B[i * dataSet.m + j]);
		}
		putchar('\n');
	}

	printf("[-] Matrix C\n");
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			printf("%-8d", dataSet.C[i * dataSet.m + j]);
		}
		putchar('\n');
	}
}

void closeDataSet(DataSet dataSet) {
	free(dataSet.A);
	free(dataSet.B);
	free(dataSet.C);
}

void add(DataSet dataSet) {
	#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < dataSet.n; i++) {
		for (int j = 0; j < dataSet.m; j++) {
			dataSet.C[i * dataSet.m + j] = dataSet.A[i * dataSet.m + j] + dataSet.B[i * dataSet.m + j];
		}
	}
}
