#include "mat.h"

#define NUM_EXP 10

int main()
{
#ifndef _OPENMP
    printf("OpenMP is not supported, sorry!\n");
    getchar();
    return 0;
#endif

    double average_running_time = 0.0;

    omp_set_num_threads(NUM_THREADS);

    short **A;
    short **B;
    short **C;

    short **A_t;
    short **C_t;

    A = allocate_matrix();
    fill_with_random_numbers(A);
    B = allocate_matrix();
    fill_with_random_numbers(B);
    C = allocate_matrix();
    fill_with_random_numbers(C);

    A_t = allocate_matrix();
    C_t = allocate_matrix();

    mat_transpose(A, A_t);

    mat_transpose(C, C_t);

    short **R;
    short **R1;
    short **R2;
    short **R3;

    for (int i = 1; i <= NUM_EXP; i++)
    {

        R = allocate_matrix();
        R1 = allocate_matrix();
        R2 = allocate_matrix();
        // R3 = allocate_matrix();

        double start = omp_get_wtime();




        // mat_mat(A_t, A, R1);

        // mat_mat(B, A, R2);

        // mat_sum(R1 , R2 , R3);

        // mat_mat(R3 , C_t , R);
// /////////////////////////////////////////

        
        mat_sum(A_t, B, R1);

        mat_mat(R1, A, R2);

        mat_mat(R2, C_t, R);


        double end = omp_get_wtime();

        double elapsed_time = end - start;

        printf("running time for %d*%d matrixes with %d threads : %f\n", MATRIX_SIZE, MATRIX_SIZE, NUM_THREADS, elapsed_time);

        average_running_time += elapsed_time;

        free_array(R);
        free_array(R1);
        free_array(R2);
        // free_array(R3);
    }
    printf("average running time : %f \n", average_running_time / NUM_EXP);

    return 0;
}

