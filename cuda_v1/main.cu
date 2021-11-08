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

void writeToFile(word16 *md5Hash, unsigned int num, const char *filename){
	FILE *outfile = NULL;
	char *outFileName = NULL;

	char *outDir = "md5text";
	char *appendStr = "-md5";
	char *inFileName = (char*)filename;
	char *token = strsep(&inFileName, "/");
	token = strsep(&inFileName, "/");
	outFileName = malloc (strlen(outDir) + strlen(token) + strlen(appendStr) + 2);
	if(outFileName){
		outFileName[0] = '\0';
		strcat(outFileName,outDir);
		strcat(outFileName,"/");
		strcat(outFileName,token);
		strcat(outFileName,appendStr);
	}

	if ((outfile = fopen (outFileName, "w")) == NULL){
		printf ("%s can't be opened for writing\n", outFileName);
		exit(0);
	}

	unsigned int i=0;
	uint8_t result[16];
	while(i<num){
		memcpy(result,md5Hash[i].word, 16);
		unsigned int j=0;
		while(j<16){
			fprintf(outfile,"%02x", result[j]);
			//printf("%02x", result[j]);
			++j;
		}
		fprintf(outfile,"\n");
		++i;
	}
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

    startTime(&devicealloctime);
    password *device_password_array;
    cudaMalloc((void**)&device_password_array, num_pwd*sizeof(password));
    cudaMemcpy(device_password_array, pwd, num_pwd*sizeof(password), cudaMemcpyHostToDevice);
    stopTime(&devicealloctime);
    printf("Device Allocation time: %f s\n", elapsedTime(devicealloctime));

    startTime(&MD5time);
    startTime(&Dicttime);
    int found = 0;

    // First we start dictionary
    for(int i = -1; found == 0 && i <= 251; ++i){
        // Copy found to cuda
        int* cuda_found;
        cudaMalloc((void**)&cuda_found, sizeof(int));
        cudaMemcpy(cuda_found, found, sizeof(int), cudaMemcpyHostToDevice);

        // Allocate size
        dim3 DimBlock(1024,1,1);
        dim3 DimGrid(num_pwd/1024, 1, 1);
        if(num_pwd%1024) DimGrid.x ++;

        // Run kernel
        mutate_and_check<<<DimGrid, DimBlock>>>(device_password_array, unsigned int num_pwd, int i, cuda_found);
        cudaDeviceSynchronize();

        cudaMemcpy(found, cuda_found, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(cuda_found);
    }
    stopTime(&Dicttime);
    printf("Dictionary Calculation Time: %f s\n", elapsedTime(Dicttime));

    if(!found){
        printf("Couldn't find the password with dictionary manipulation\n");
        for(int i = 1; i <= 5; ++i){
            size_t search_space = pow(26, i);

            // Copy found to cuda
            int* cuda_found;
            cudaMalloc((void**)&cuda_found, sizeof(int));
            cudaMemcpy(cuda_found, found, sizeof(int), cudaMemcpyHostToDevice);

            // Allocate size
            dim3 DimBlock(1024,1,1);
            dim3 DimGrid(search_space/1024, 1, 1);
            if(search_space%1024) DimGrid.x ++;

            // Run kernel
            brute_force<<<DimGrid, DimBlock>>>(search_space, cuda_found);
            cudaDeviceSynchronize();

            cudaMemcpy(found, cuda_found, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(cuda_found);
        }
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