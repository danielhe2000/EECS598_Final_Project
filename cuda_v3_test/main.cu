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
    if (argc < 2) {
        printf("usage: %s 'num of cases to be tested'\n", argv[0]);
        return 1;
    }

    int num_of_tests = atoi(argv[1]);

    // Initiate timer
    Timer totaltimer,filereadtimer, devicealloctime, MD5time, Dicttime;
    startTime(&totaltimer);

    // Read in dictionary
    const char *filename = "plaintext/mostcommon-10k";

    FILE *infile;
    if ((infile = fopen (filename, "r")) == NULL){
		printf ("%s can't be opened\n", filename);
		exit(0);
	}

    startTime(&filereadtimer);
    unsigned int num_pwd;
    password *pwd;
    readPwdFromFile(infile, &pwd, &num_pwd);
    printf("Total Dictionary Words: %d\n",num_pwd);
    //printall(pwd, numPwd);
    stopTime(&filereadtimer);
    printf("File read time: %f s\n", elapsedTime(filereadtimer));

    // Read in test case
    const char *testcase_filename = "../Testcase/5000_md5.txt";
    FILE *testfile;

    if ((testfile = fopen (testcase_filename, "r")) == NULL){
		printf ("%s can't be opened\n", testcase_filename);
		exit(0);
	}
    int all_passed = 1;
    int test_index;
    for(test_index = 0; test_index < num_of_tests; test_index++){
        printf("##################################################\n");

        // to be tested
        char *test = NULL;
        uint8_t int_test[16];
        
	    size_t len = 0;
	    int read_len = 0;
        read_len = getline(&test, &len, testfile);

        hashToUint8(test,int_test);

        // Initiate for 
        init_dictionary_seq();
        init_md5_const();
        init_target_hash_dictionary(int_test);
        init_target_hash_brute_force(int_test);

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
        }

        // iterate checking
        for (int i = 0; i < count; ++i) {
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

        if(!found){
            printf("Couldn't find the password with dictionary manipulation\n");
            for(unsigned int i = 1; !found && i <= MAX_LETTERS; ++i){
                // parallel this 
                size_t search_space = pow(26, i);
                // printf(">>>>>>>>>>>>>>>>Search space %d<<<<<<<<<<<<<<<\n", search_space);
                // Copy found to cuda
                int* cuda_found;
                cudaMalloc((void**)&cuda_found, sizeof(int));
                cudaMemcpy(cuda_found, &found, sizeof(int), cudaMemcpyHostToDevice);

                // Allocate size
                dim3 DimBlock(1024,1,1);
                dim3 DimGrid(search_space/1024, 1, 1);
                if(search_space%1024) DimGrid.x ++;
                
                // Run kernel
                brute_force<<<DimGrid, DimBlock>>>(i, cuda_found);
                // printf("Possible error: %s\n", cudaGetErrorString(cudaGetLastError()));
                cudaDeviceSynchronize();

                cudaMemcpy(&found, cuda_found, sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(cuda_found);
            }
        }
        if(!found){
            printf("Sorry. Couldn't find the password of test number %d\n", test_index);
            all_passed = 0;
        }
        else{
            printf("Test number %d succeeded\n", test_index);
        }
        stopTime(&MD5time);
        printf("MD5 Calculation time: %f s\n", elapsedTime(MD5time));

        stopTime(&totaltimer);
        printf("Total Time: %f s\n", elapsedTime(totaltimer));

        cudaFree(device_password_array);
        delete [] streams;
    }

    if(!all_passed){
        printf(">>>>>>>>>>> One of the test cases went wrong !!!!!!!!!!!!!!\n");
    }
    else{
        printf(">>>>>>>>>>> Congrats! All cases passed !!!!!!!!!!!!!!\n");
    }

    printf("Total Time: %f s\n", elapsedTime(totaltimer));
    

    // Clean up
	if(infile != NULL)	fclose (infile);
    if(testfile != NULL) fclose (testfile);
    

    return 0;
}