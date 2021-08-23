#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MATRIX_SIZE 512
#define BLOCK_NUM 4
#define NUM_THREADS 2
#define NUM_EXP 10

void LUDecompose(float **a, float **l, float **u, int size);
float **invers_matrix(float **A, int size);
float **allocate_float_matrix(int size);
void fill_with_random_float_numbers(float **mat, int size);
void mat_diff_float(float **A, float **B, float **C, int size);
void mat_mat_float(float **A, float **B, float **C, int size);
void print_float_matrix(float **A, int size);
void free_float_array(float **A, int size);
float **create_augmented_matrix(float **A, int size);
float **copy(float **A, int size);
void fill_with_zero(float **mat, int size);

int main()
{
#ifndef _OPENMP
    printf("OpenMP is not supported, sorry!\n");
    getchar();
    return 0;
#endif
    double time_sum = 0;

    for(int ex = 0 ; ex < NUM_EXP ; ex++)
    {    
        float **A[BLOCK_NUM][BLOCK_NUM];
        float **L[BLOCK_NUM][BLOCK_NUM];
        float **U[BLOCK_NUM][BLOCK_NUM];

        int block_size = MATRIX_SIZE / BLOCK_NUM;

        for (int i = 0; i < BLOCK_NUM; i++)
        {
            for (int j = 0; j < BLOCK_NUM; j++)
            {
                A[i][j] = allocate_float_matrix(block_size);
                fill_with_random_float_numbers(A[i][j], block_size);
            }
        }

        for (int i = 0; i < BLOCK_NUM; i++)
        {
            for (int j = 0; j < BLOCK_NUM; j++)
            {
                L[i][j] = allocate_float_matrix(block_size);
                fill_with_zero(L[i][j], block_size);
            }
        }

        for (int i = 0; i < BLOCK_NUM; i++)
        {
            for (int j = 0; j < BLOCK_NUM; j++)
            {
                U[i][j] = allocate_float_matrix(block_size);
                fill_with_zero(U[i][j], block_size);
            }
        }

        float **U_inv[BLOCK_NUM];
        float **L_inv[BLOCK_NUM];

        for (int i = 0; i < BLOCK_NUM; i++)
        {
            U_inv[i] = allocate_float_matrix(block_size);
        }

        for (int i = 0; i < BLOCK_NUM; i++)
        {
            L_inv[i] = allocate_float_matrix(block_size);
        }

        double start = omp_get_wtime();

        for (int step = 0; step < BLOCK_NUM; step++)
        {
            #pragma omp parallel num_threads(NUM_THREADS)
            {
                #pragma omp single
                {
                    LUDecompose(A[step][step], L[step][step], U[step][step], block_size);

                    #pragma omp task
                    {
                        U_inv[step] = invers_matrix(U[step][step], block_size);
                        for (int i = step + 1; i < BLOCK_NUM; i++)
                            mat_mat_float(A[i][step], U_inv[step], L[i][step], block_size);
                    }

                    #pragma omp task
                    {
                        L_inv[step] = invers_matrix(L[step][step], block_size);
                        for (int i = step + 1; i < BLOCK_NUM; i++)
                            mat_mat_float(L_inv[step], A[step][i], U[step][i], block_size);
                    }
                }

                #pragma omp single
                {

                    for (int i = step + 1; i < BLOCK_NUM; i++)
                    {
                        for (int j = step + 1; j < BLOCK_NUM; j++)
                        {
                            #pragma omp task
                            {
                                float **R = allocate_float_matrix(block_size);
                                mat_mat_float(L[i][step], U[step][j], R, block_size);
                                mat_diff_float(A[i][j], R, A[i][j], block_size);

                                free_float_array(R, block_size);
                            }
                        }
                    }
                }
            }
        }

        
        float det1 = 1.0, det2 = 1.0;
        #pragma omp for
        for (int i = 0; i < BLOCK_NUM; i++)
        {

            float **l_block = L[i][i];
            float **u_block = U[i][i];

            for (int j = 0; j < block_size; j++)
            {
                det1 *= l_block[j][j];
                det2 *= u_block[j][j];
            }
        }

        double end = omp_get_wtime() - start;

        time_sum += end;

        printf("time elapsed with %d threads : %f \n", NUM_THREADS , end);

        for (int i = 0; i < BLOCK_NUM; i++)
        {
            for (int j = 0; j < BLOCK_NUM; j++)
            {
                free_float_array(A[i][j], block_size);
            }
        }

        for (int i = 0; i < BLOCK_NUM; i++)
        {
            for (int j = 0; j < BLOCK_NUM; j++)
            {
                free_float_array(L[i][j], block_size);
            }
        }

        for (int i = 0; i < BLOCK_NUM; i++)
        {
            for (int j = 0; j < BLOCK_NUM; j++)
            {
                free_float_array(U[i][j], block_size);
            }
        }
    }

    printf("\n average : %f \n" , time_sum / NUM_EXP);

    return 0;
}

void LUDecompose(float **a, float **l, float **u, int size)
{

    for (int i = 0; i < size; i++)
    {
        for (int k = i; k < size; k++)
        {
            float sum = 0;
            for (int j = 0; j < i; j++)
                sum += (l[i][j] * u[j][k]);

            u[i][k] = a[i][k] - sum;
        }
        for (int k = 0; k < size; k++)
        {

            if (i == k)
                l[i][i] = 1;
            else
            {
                float sum = 0.0;
                for (int j = 0; j < i; j++)
                    sum += (l[k][j] * u[j][i]);

                l[k][i] = (a[k][i] - sum) / u[i][i];
            }
        }
    }
}
float **invers_matrix(float **A, int size)
{

    float **matrix = create_augmented_matrix(A, size);

    float temp;

    for (int i = size - 1; i > 0; i--)
    {
        if (matrix[i - 1][0] < matrix[i][0])
        {
            float *temp = matrix[i];
            matrix[i] = matrix[i - 1];
            matrix[i - 1] = temp;
        }
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (j != i)
            {
                temp = matrix[j][i] / matrix[i][i];
                for (int k = 0; k < 2 * size; k++)
                {
                    matrix[j][k] -= matrix[i][k] * temp;
                }
            }
        }
    }

    for (int i = 0; i < size; i++)
    {

        temp = matrix[i][i];
        for (int j = 0; j < 2 * size; j++)
        {

            matrix[i][j] = matrix[i][j] / temp;
        }
    }

    float **result = allocate_float_matrix(size);

    for (int i = 0; i < size; i++)
        for (int j = size; j < 2 * size; j++)
            result[i][j - size] = matrix[i][j];

    free_float_array(matrix, size * 2);

    return result;
}

float **allocate_float_matrix(int size)
{
    float **mat;
    mat = (float **)malloc(sizeof(*mat) * size);

    for (int i = 0; i < size; i++)
        mat[i] = (float *)malloc(sizeof(mat[i]) * size);

    return mat;
}

void fill_with_random_float_numbers(float **mat, int size)
{
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            mat[i][j] = (float)(rand() % 10 + 1) / 100;
}

void fill_with_zero(float **mat, int size)
{
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            mat[i][j] = 0;
}

void mat_diff_float(float **A, float **B, float **C, int size)
{
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void mat_mat_float(float **A, float **B, float **C, int size)
{
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
                C[i][j] += A[i][k] * B[k][j];
}

void print_float_matrix(float **A, int size)
{

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
}

void free_float_array(float **A, int size)
{

    for (int i; i < size; i++)
    {

        float *row = A[i];

        free(row);
    }
}

float **create_augmented_matrix(float **A, int size)
{

    float **_A = allocate_float_matrix(size * 2);

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            _A[i][j] = A[i][j];

    for (int i = 0; i < size; i++)
        for (int j = size; j < size * 2; j++)
        {
            if (j == (i + size))
                _A[i][j] = 1;
        }

    return _A;
}

float **copy(float **A, int size)
{

    float **_A = allocate_float_matrix(size);

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)

            _A[i][j] = A[i][j];

    return _A;
}
