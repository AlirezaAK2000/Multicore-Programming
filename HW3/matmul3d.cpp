#include <iostream>
using namespace std;

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<time.h>

#define MATRIX_SIZE 512
#define THREADS_NUM 1
#define EXP_NUM 5

typedef struct {
	short data[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE];
} Tensor3D;


Tensor3D A;
Tensor3D B;
Tensor3D C;

void fill_with_rand_nums(Tensor3D* A);
void fill_with_zero(Tensor3D* A);
void fill_matrix(Tensor3D* A, Tensor3D* B, Tensor3D* C);
void print_matrix(Tensor3D* A);

void matmul3d_block_p(Tensor3D* A, Tensor3D* B, Tensor3D* C);
void matmul3d_row_p(Tensor3D* A, Tensor3D* B, Tensor3D* C);
void matmul3d_col_p(Tensor3D* A, Tensor3D* B, Tensor3D* C);

void free_matrix(Tensor3D* A);

int main()
{

		// checks if openMP is available
	#ifndef _OPENMP
		printf("OpenMP is not supported, sorry!\n");
		getchar();
		return 0;
	#endif 

	omp_set_num_threads(THREADS_NUM);
	printf("number of threads : %d\n" , THREADS_NUM);

	double exp_times_sum = 0;
	fill_matrix(&A, &B, &C);

	for (int i = 0; i < EXP_NUM; i++)
	{
		double start = omp_get_wtime();
		matmul3d_col_p(&A, &B, &C);
		double end = omp_get_wtime() - start;
		printf("elapsed time : %f exp : %d\n", end, i + 1);
		exp_times_sum += end;
	}

	printf("average elapsed time for block parallelism: %f\n", exp_times_sum / EXP_NUM);


	// free_matrix(&A);
	// free_matrix(&B);
	// free_matrix(&C);

	return 0;

}



void fill_with_rand_nums(Tensor3D* A) {
	for (int i = 0; i < MATRIX_SIZE; i++) {
		for (int j = 0; j < MATRIX_SIZE; j++) {
			for (int k = 0; k < MATRIX_SIZE; k++) {
				A->data[i][j][k] = rand() % 10;
			}
		}
	}
}

void fill_with_zero(Tensor3D* A) {
	for (int i = 0; i < MATRIX_SIZE; i++) {
		for (int j = 0; j < MATRIX_SIZE; j++) {
			for (int k = 0; k < MATRIX_SIZE; k++) {
				A->data[i][j][k] = 0;
			}
		}
	}
}


void fill_matrix(Tensor3D* A, Tensor3D* B, Tensor3D* C) {
	double int_size = sizeof(short);
	double size_mb = ((double)(MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE)) * int_size / (1024.0 * 1024.0);
	printf(" size of matrixes : %lf MB\n", size_mb);
	fill_with_rand_nums(A);
	fill_with_rand_nums(B);
	fill_with_zero(C);
}

void free_matrix(Tensor3D* A) {
	free(A->data);
}

void print_matrix(Tensor3D* A) {
	printf("[");
	for (int i = 0; i < MATRIX_SIZE; i++) {
		printf("[\n");
		for (int j = 0; j < MATRIX_SIZE; j++) {
			for (int k = 0; k < MATRIX_SIZE; k++) {
				printf("%d ", A->data[i][j][k]);
			}
			printf("\n");
		}
		printf("]\n");
	}
	printf("]\n");
}


void matmul3d_block_p(Tensor3D* A, Tensor3D* B, Tensor3D* C) {
	#pragma omp parallel for 
	for (int i = 0; i < MATRIX_SIZE; i++) {
		for (int j = 0; j < MATRIX_SIZE; j++) {
			for (int k = 0; k < MATRIX_SIZE; k++) {
				for (int p = 0; p < MATRIX_SIZE; p++) {
					C->data[i][j][k] += A->data[i][j][p] * B->data[i][p][k];
				}
			}
		}
	}
}

void matmul3d_row_p(Tensor3D* A, Tensor3D* B, Tensor3D* C) {
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < MATRIX_SIZE; i++) {
		for (int j = 0; j < MATRIX_SIZE; j++) {
			for (int k = 0; k < MATRIX_SIZE; k++) {
				for (int p = 0; p < MATRIX_SIZE; p++) {
					C->data[i][j][k] += A->data[i][j][p] * B->data[i][p][k];
				}
			}
		}
	}
}



void matmul3d_col_p(Tensor3D* A, Tensor3D* B, Tensor3D* C) {
	#pragma omp parallel for collapse(3)
	for (int i = 0; i < MATRIX_SIZE; i++) {
		for (int j = 0; j < MATRIX_SIZE; j++) {
			for (int k = 0; k < MATRIX_SIZE; k++) {
				for (int p = 0; p < MATRIX_SIZE; p++) {
					C->data[i][j][k] += A->data[i][j][p] * B->data[i][p][k];
				}
			}
		}
	}
}