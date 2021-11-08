#include "dictionary.h"

__constant__ password seq[20];
__constant__ uint8_t target_hash[16];

void init_dictionary_seq(){
    cpu_seq = (password *)malloc(20*sizeof(password));

    memcpy(cpu_seq[0].word,"123", 3); cpu_seq[0].length = 3;
    memcpy(cpu_seq[1].word,"1234", 4); cpu_seq[1].length = 4;
    memcpy(cpu_seq[2].word,"12345", 5);   cpu_seq[2].length = 5;
    memcpy(cpu_seq[3].word,"123456", 6);   cpu_seq[3].length = 6;
    memcpy(cpu_seq[4].word,"1234567", 7);   cpu_seq[4].length = 7;
    memcpy(cpu_seq[5].word,"12345678", 8);   cpu_seq[5].length = 8;
    memcpy(cpu_seq[6].word,"123456789", 9);   cpu_seq[6].length = 9;
    memcpy(cpu_seq[7].word,"1234567890", 10);   cpu_seq[7].length = 10;
    memcpy(cpu_seq[8].word,"696969", 6);   cpu_seq[8].length = 6;
    memcpy(cpu_seq[9].word,"111111", 6);   cpu_seq[9].length = 6;
    memcpy(cpu_seq[10].word,"1111", 4);   cpu_seq[10].length = 4;
    memcpy(cpu_seq[11].word,"1212", 4);   cpu_seq[11].length = 4;
    memcpy(cpu_seq[12].word,"7777", 4);   cpu_seq[12].length = 4;
    memcpy(cpu_seq[13].word,"1004", 4);   cpu_seq[13].length = 4;
    memcpy(cpu_seq[14].word,"2000", 4);   cpu_seq[14].length = 4;
    memcpy(cpu_seq[15].word,"4444", 4);   cpu_seq[15].length = 4;
    memcpy(cpu_seq[16].word,"2222", 4);   cpu_seq[16].length = 4;
    memcpy(cpu_seq[17].word,"6969", 4);   cpu_seq[17].length = 4;
    memcpy(cpu_seq[18].word,"9999", 4);   cpu_seq[18].length = 4;
    memcpy(cpu_seq[19].word, "3333", 4);   cpu_seq[19].length = 4;

    cudaMemcpyToSymbol(seq, seq_cpu, sizeof(password)*20);  
    free(cpu_seq);
}

void init_target_hash_dictionary(uint8_t* int_test){
    cudaMemcpyToSymbol(target_hash, int_test, sizeof(uint8_t)*16);  
}

__global__ void mutate_and_check(password *dict, unsigned int numwords, int mutation_method, int* found){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= numwords) return;

    // First, try to copy the password from memory
    password new_pas;
    new_pas.length = dict[index].length;
    for(int i = 0; i < new_pas.length; ++i){
        new_pas.word[i] = dict[index].word[i];
    }

    // Then, try to mutate the input password
    if (mutation_method==0){
        /* First letter uppercase */
        if (new_pas.word[0] >= 'a' && new_pas.word[0] <= 'z')
            new_pas.word[0] +='A'-'a';
    }
    else if (mutation_method==1){
        /* Last letter uppercase */
        size_t len = new_pas.length;
        if (new_pas.word[len-1] >= 'a' && new_pas.word[len-1] <= 'z')
            new_pas.word[len-1] += 'A'-'a';
    }
    else if (mutation_method>=2 && mutation_method<=11){
        /* Add one digit to end
         * iterator: z-2    */
        size_t len = new_pas.length;
        new_pas.word[len] = '0' + z-2;
        new_pas.length += 1;
    }
   /* Add sequence of numbers at end; e.g. 1234, 84, 1999 */
    else  if (mutation_method>=12 && mutation_method<=111){
        // 0 to 99
        // iterator: z-12
        size_t len = new_pas.length;
        new_pas.word[len] = '0' + ((z-12)/10)%10;
        new_pas.word[len+1] = '0' + (z-12)%10;
        new_pas.length += 2;
    }
    else if (mutation_method>=112 && mutation_method<=231){
        // 1900 to 2020
        // iterator: z + (1900-112)
        size_t len = new_pas.length;
        new_pas.word[len] = '0' + ((z+1900-112)/1000)%10;
        new_pas.word[len+1] = '0' + ((z+1900-112)/100)%10;
        new_pas.word[len+2] = '0' + ((z+1900-112)/10)%10;
        new_pas.word[len+3] = '0' + (z+1900-112)%10;
        new_pas.length += 4;
    }
    else if (mutation_method>=232 && mutation_method<=251){
        // Other common sequences
        // iterator: z-232
        //sprintf(&temp,"%s",sequences[z-252]);
        size_t len = new_pas.length;
        for(int i = 0; i < seq[z-232].length; ++i){
            new_pas.word[len] + i = seq[z-232].word[i];
        }
        new_pas.length = len + seq[z-232].length;
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