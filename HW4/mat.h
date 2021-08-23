#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

#define MATRIX_SIZE 64

#define NUM_THREADS 1

short **allocate_matrix()
{
    short **mat;
    mat = (short **)malloc(sizeof(*mat) * MATRIX_SIZE);

    for (int i = 0; i < MATRIX_SIZE; i++)
        mat[i] = (short *)malloc(sizeof(mat[i]) * MATRIX_SIZE);

    return mat;
}

void fill_with_random_numbers(short **mat)
{
    for (int i = 0; i < MATRIX_SIZE; i++)
        for (int j = 0; j < MATRIX_SIZE; j++)
            mat[i][j] = rand() % 10;
}


void mat_sum(short **A, short **B, short **C)
{
#ifndef _OPENMP
    printf("OpenMP is not supported, sorry!\n");
#endif

#pragma omp parallel for
    for (int i = 0; i < MATRIX_SIZE; i++)
        for (int j = 0; j < MATRIX_SIZE; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void mat_mat(short **A, short **B, short **C)
{

#ifndef _OPENMP
    printf("OpenMP is not supported, sorry!\n");
#endif

#pragma omp parallel for
    for (int i = 0; i < MATRIX_SIZE; i++)
        for (int j = 0; j < MATRIX_SIZE; j++)
            for (int k = 0; k < MATRIX_SIZE; k++)
                C[i][j] += A[i][k] * B[k][j];
}

void mat_transpose(short **mat, short **mat_t)
{
#ifndef _OPENMP
    printf("OpenMP is not supported, sorry!\n");
#endif


#pragma omp parallel for
    for (int i = 0; i < MATRIX_SIZE; i++)
        for (int j = 0; j < MATRIX_SIZE; j++)
            mat_t[i][j] = mat[j][i];
}

void print_matrix(short **A)
{

    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            printf("%d ", A[i][j]);
        }
        printf("\n");
    }
}

void free_array(short **A)
{

    for (int i; i < MATRIX_SIZE; i++)
    {

        short *row = A[i];

        free(row);
    }
}








// ////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////








