#ifndef __GPU_MD5__
#define __GPU_MD5__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>

typedef struct word16_t{
    char word[16];
} word16;

typedef struct password_t{
    char word[56];
    size_t length;
} password;

__device__ void md5(password *, uint8_t *digest);

void init_md5_const();

#endif