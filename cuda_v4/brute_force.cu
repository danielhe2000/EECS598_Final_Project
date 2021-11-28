#include "brute_force.h"

extern __constant__ uint8_t target_hash[16];

void init_target_hash_brute_force(uint8_t* int_test){
    cudaMemcpyToSymbol(target_hash, int_test, sizeof(uint8_t)*16);  
}

__global__ void brute_force(size_t offset, int* found){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x + offset;
    // if(index >= (int) (pow(26, pas_length))) return;
    // printf("Index: %d\n", index);
    // printf("Password length: %d\n", pas_length);
    password new_pas;
    memset(&new_pas,0,sizeof(password));
    bool ctn = 1;
    
    for(int i = 0; i < MAX_LETTERS; ++i){
        new_pas.word[i] = 'a' * ctn + (uint8_t) (index%NUM_LETTERS);
        new_pas.length += ctn;
        index /= NUM_LETTERS;
        ctn = (index != 0);
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