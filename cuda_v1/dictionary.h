#ifndef __GPU_DICT_MANI
#define __GPU_DICT_MANI
#include "md5.h"

void init_dictionary_seq();
void init_target_hash_dictionary(uint8_t* int_test);

// When the word is found, this method increment "found" by 1
__global__ void mutate_and_check(password *dict, unsigned int numwords, int mutation_method, int* found);

#endif