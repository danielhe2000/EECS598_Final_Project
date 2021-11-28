#ifndef __BRUT_FORC
#define __BRUT_FORC

#define MAX_LETTERS	6
#define NUM_LETTERS	26
#include "md5.h"

__global__ void brute_force(unsigned int pas_length, int* found);
void init_target_hash_brute_force(uint8_t* int_test);


#endif