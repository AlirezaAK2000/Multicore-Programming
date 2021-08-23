#include <omp.h>
#include <stdio.h>
#include <stdlib.h>



#define NUM_RESOURCES 8
#define ATTEMPTS 4


omp_lock_t locks[NUM_RESOURCES];


void calc(int i , int j){
    int thread_num = omp_get_thread_num();

    printf("threads %d wants %d and %d \n" , thread_num , i , j);

    omp_set_lock(&locks[i]);
    omp_set_lock(&locks[j]);

    int k = 0;

    for(int p = 0 ; p < 1000000 ; p++){
        k++;
    }

    omp_unset_lock(&locks[i]);
    omp_unset_lock(&locks[j]);

}


int main(){
#ifndef _OPENMP
    printf("OpenMP is not supported, sorry!\n");
    getchar();
    return 0;
#endif

    for (size_t i = 0; i < NUM_RESOURCES; i++)
        omp_init_lock(&locks[i]);

    for (int k = 0; k < ATTEMPTS; k++)
    {
        printf("attempt %d : \n\n" , k + 1);
        #pragma omp parallel num_threads(NUM_RESOURCES)
        {
            int i = rand() % NUM_RESOURCES;
            int j = rand() % NUM_RESOURCES;
            if(i == j)
                j = (j + 1 ) % NUM_RESOURCES;

            #pragma omp barrier
            calc(i , j);
        }
    }

    for (size_t i = 0; i < NUM_RESOURCES; i++)
        omp_destroy_lock(&locks[i]);

    return 0;
}