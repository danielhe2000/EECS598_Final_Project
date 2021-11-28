CC 			= gcc
DEBUG 		= -g
CFLAGS		= -c -Wall $(DEBUG) -O3
LFLAGS		= $(DEBUG)
DEPEND		= support.o
EXE			= md5
OBJ			= $(DEPEND) md5.o main.o dictman.o brute_force.o

$(EXE): $(OBJ)
	$(CC) $(OBJ) -o $(EXE)

md5%.o: %.c
	$(CC) $(CFLAGS) -o $@ $<
	
dictman.o: dictman.c dictman.h
	$(CC) $(CFLAGS) -o $@ dictman.c
	
brute_force.o: brute_force.c brute_force.h
	$(CC) $(CFLAGS) -o $@ brute_force.c
			
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
