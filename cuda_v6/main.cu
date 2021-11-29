#include "brute_force.h"
#include "dictionary.h"
#include "support.h"
#include "md5.h"
#include <math.h>

void readPwdFromFile(FILE *infile, password **pwd, unsigned int *numLines){

	unsigned int numberOfLines = 0;
	int ch;
	while (EOF != (ch=getc(infile))){
    	if (ch=='\n'){
    		++numberOfLines;
	    	if(numberOfLines == (UINT_MAX/sizeof(char*)))
	    		break;
    	}
    }
    rewind(infile);

    *pwd = (password*)malloc(numberOfLines*sizeof(password));
    memset(*pwd, 0, numberOfLines*sizeof(password)); // reset memory
    if(*pwd == NULL){
        printf("\nERROR: Memory allocation did not complete successfully! Exiting.");
        exit(0);
    }

    char *line = NULL;
	size_t len = 0;
	int read_len = 0;
	unsigned int i=0;
	unsigned int toReduce = 0;
	while (i<numberOfLines) {
		read_len = getline(&line, &len, infile);
		if(read_len != -1){
			if(line[read_len-1] == '\n')    read_len = read_len - 1;
			if(line[read_len-1] == '\r')    read_len = read_len - 1;

			if(read_len > 45){
                //printf("Skipping (too big) - %s\n",line);
                ++toReduce;
            } else {
                // (*pwd)[i-toReduce] = (char*)malloc( (read_len+1)*sizeof(char));
                memcpy((*pwd)[i-toReduce].word,line,read_len);
                (*pwd)[i-toReduce].length = read_len;
                //printf("Pwd Read: %s, %d\n", (*pwd)[i], read_len);
	  		}
	  	} else {
            ++toReduce;
	  	}
        free(line);
        line = NULL;
		len = 0;
	  	i++;
	}
	*numLines = numberOfLines-toReduce;
	//passwd = &pwd;
}

inline void printpwd(password *pwd){
    unsigned int i=0;
    char *str = pwd->word;
    while(i < pwd->length) {
        printf("%c",str[i]);
		++i;
	}
}


void printall(password *pwd, unsigned int num){
	unsigned int i=0;
	while(i<num) {
		//char *str = pwd[i];
		printf("Pwd as Stored: ");
        printpwd(&(pwd[i]));
		printf("\n");
		++i;
	}
	printf("Num of lines : %d\n",num);
}

void hashToUint8(char *charHash, uint8_t intHash[]){
    char tempChar[16][3];
    int j=0;
    while(j<16){
        tempChar[j][0] = charHash[j*2];
        tempChar[j][1] = charHash[j*2+1];
        tempChar[j][2] = '\0';
        ++j;
    }
    j = 0;
    while(j<16){
        sscanf(tempChar[j], "%x", (unsigned int*)(&(intHash[j])));
        ++j;
    }
}

int main(int argc, char **argv){
    // to be tested
    char *test;
    uint8_t int_test[16];
    if (argc < 2) {
        printf("usage: %s 'stringhash'\n", argv[0]);
        return 1;
    }

    test = argv[1];
    if(strlen(test) != 32){
        printf ("Invalid hash. Exiting.\n");
		exit(0);
    }

    hashToUint8(test,int_test);

    // Initiate for 
    init_dictionary_seq();
    init_md5_const();
    init_target_hash_dictionary(int_test);
    init_target_hash_brute_force(int_test);

    const char *filename = "plaintext/mostcommon-10k";

    FILE *infile;
    if ((infile = fopen (filename, "r")) == NULL){
		printf ("%s can't be opened\n", filename);
		exit(0);
	}

    Timer totaltimer,filereadtimer, devicealloctime, MD5time, Dicttime;
    startTime(&totaltimer);

    startTime(&filereadtimer);
    unsigned int num_pwd;
    password *pwd;
    readPwdFromFile(infile, &pwd, &num_pwd);
    printf("Total Dictionary Words: %d\n",num_pwd);
    //printall(pwd, numPwd);
    stopTime(&filereadtimer);
    printf("File read time: %f s\n", elapsedTime(filereadtimer));

    int count = num_pwd/BLOCKTHREADS;
    if (num_pwd%BLOCKTHREADS) count ++;
    cudaStream_t * streams = new cudaStream_t[count]; // init streams
    for (int i = 0; i < count; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    startTime(&devicealloctime);
    password *device_password_array;
    cudaMalloc((void**)&device_password_array, num_pwd*sizeof(password));

    // streaming memcpy, each stream takes 512 pwds
    for (int i = 0; i < count; ++i) {
        int numwords = BLOCKTHREADS;
        if ((i+1) * BLOCKTHREADS > num_pwd) numwords = num_pwd%BLOCKTHREADS; 
        cudaMemcpyAsync(device_password_array + i*BLOCKTHREADS, pwd + i*BLOCKTHREADS, numwords*sizeof(password), cudaMemcpyHostToDevice, streams[i]);
    }

    stopTime(&devicealloctime);
    printf("Device Allocation time: %f s\n", elapsedTime(devicealloctime));

    startTime(&MD5time);
    startTime(&Dicttime);

    dim3 DimBlock(BLOCKTHREADS, 1, 1);
    dim3 DimGrid(253, 1, 1);

    int found = 0;
    int* cuda_found;
    cudaMalloc((void**)&cuda_found, sizeof(int));
    cudaMemcpy(cuda_found, &found, sizeof(int), cudaMemcpyHostToDevice);

    // streaming processing, each kernel takes 512 pwds and 253 mutations
    for (int i = 0; i < count; ++i) {
        unsigned int numwords = BLOCKTHREADS;
        if ((i+1) * BLOCKTHREADS > num_pwd) numwords = num_pwd%BLOCKTHREADS; 
        uint *pwd_data = (uint *) (device_password_array + i*BLOCKTHREADS);
        mutate_and_check<<<DimGrid, DimBlock, 0, streams[i]>>>(pwd_data, numwords, cuda_found);
        // printf("Possible error: %s\n", cudaGetErrorString(cudaGetLastError()));
        cudaStreamSynchronize(streams[i]);
        cudaMemcpy(&found, cuda_found, sizeof(int), cudaMemcpyDeviceToHost);
        if (found) {
            for (; i < count; ++i) {
                cudaStreamDestroy(streams[i]);
            }
            break;
        }
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(cuda_found);
    
    stopTime(&Dicttime);
    printf("Dictionary Calculation Time: %f s\n", elapsedTime(Dicttime));

    delete [] streams;

    if(!found){
        printf("Couldn't find the password with dictionary manipulation\n");
        // total = 8353082582 = 26 + 26^2 + ... + 26^7
        size_t search_space = NUM_LETTERS;
        for (int i = 0; i < MAX_LETTERS-1; ++i) {
            search_space = NUM_LETTERS + search_space * NUM_LETTERS;
        }
        printf("max length of brute force: %d\n", MAX_LETTERS);
        printf("total possibilities: %ld\n", search_space);
        size_t count = search_space/PARTITION;
        if (search_space%PARTITION) count++; // 996 streams when MAX_LETTERS is 7

        // int count = 996;
        printf("total cycles: %ld\n", count);

        int* cuda_found;
        cudaMalloc((void**)&cuda_found, sizeof(int));
        cudaMemcpy(cuda_found, &found, sizeof(int), cudaMemcpyHostToDevice);

        // Allocate size
        dim3 DimBlock(1024, 1, 1);
        dim3 DimGrid(8192, 1, 1);

        size_t offset = 0; // increment by PARTITION every iter

        for (size_t i = 0; i < count; ++i) {
            brute_force<<<DimGrid, DimBlock>>>(offset, cuda_found);
            offset += PARTITION;
            // printf("Possible error: %s\n", cudaGetErrorString(cudaGetLastError()));
            cudaDeviceSynchronize();
            cudaMemcpy(&found, cuda_found, sizeof(int), cudaMemcpyDeviceToHost);
            if (found) {
                break;
            }
        }

        cudaFree(cuda_found);
    }
    if(!found){
        printf("Sorry. Couldn't find the password\n");
    }
    stopTime(&MD5time);
    printf("MD5 Calculation time: %f s\n", elapsedTime(MD5time));

    stopTime(&totaltimer);
    printf("Total Time: %f s\n", elapsedTime(totaltimer));

    // Clean up
	if(infile != NULL)	fclose (infile);
    cudaFree(device_password_array);

    return 0;
}