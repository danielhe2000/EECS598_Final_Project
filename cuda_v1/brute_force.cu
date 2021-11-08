#include "brute_force.h"

__constant__ uint8_t target_hash[16];

void init_target_hash_brute_force(uint8_t* int_test){
    cudaMemcpyToSymbol(target_hash, int_test, sizeof(uint8_t)*16);  
}

__global__ void brute_force(unsigned int pas_length, int* found){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    password new_pas;
    new_pas.length = pas_length;
    for(int i = 0; i < pas_length; ++i){
        new_pas[i] = 'a' + index%26;
        index = index / 26;
    }
    
    word16 md5_hash;
    md5(&new_pas, (uint8_t*)md5_hash.word);
    int flag = 1;
    unsigned int j=0;
    while(j<16){
        if(target_hash[j] != (uint8_t)md5_hash.word[j]){
            flag = 0;
            break;
        }
        ++j;
    }
    if(flag == 1){
        atomicAdd(found, 1);
        printf("\n!!!!PASSWORD FOUND!!!!\nPassword is: ");
        for(int i = 0; i < new_pas.length; ++i){
            printf("%c", new_pas.word[i]);
        }
        printf("\n");
    }
}