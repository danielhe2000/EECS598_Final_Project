CC 			= nvcc
CFLAGS		= -rdc=true 
EXE			= md5


$(EXE): brute_force.cu dictionary.cu main.cu md5.cu support.cu
	$(CC) $(CFLAGS) brute_force.cu dictionary.cu main.cu md5.cu support.cu -o $(EXE)
			
wiki: md5_wiki.c
	gcc -o wiki md5_wiki.c

default: $(EXE)

cleangem:
	rm -rf *.o*
	rm -rf *.e*

clean:
	rm -rf *.o
	rm -rf $(EXE)
	rm -rf wiki
