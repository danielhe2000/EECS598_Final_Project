#ifndef __BRUT_FORC
#define __BRUT_FORC

#define MAX_LETTERS	13
#define NUM_LETTERS	26
// #define SEARCH_SPACE 8353082582
#define PARTITION 8388608  // 2^23 = 2^13 * 2^10 = 8192 * 1024
#include "md5.h"

__global__ void brute_force(size_t offset, int* found);
void init_target_hash_brute_force(uint8_t* int_test);


#endif