CC = gcc
CFLAGS = -O3 -funroll-all-loops -march=native
CPP_FLAGS = 
LIBS = -lm

ifndef NO_OMP
	CPP_FLAGS += -fopenmp
endif
ifdef BLAS
	LIBS += -lblas
	CPP_FLAGS += -DHAVE_BLAS
endif

NPROCS = $(shell grep -c 'processor' /proc/cpuinfo)
MAKEFLAGS += -j$(NPROCS)


all: arr.o mm.o network.o main.c
	$(CC) -o nnic arr.o mm.o network.o main.c $(CFLAGS) $(CPP_FLAGS) $(LIBS)

network.o: network.c
	$(CC) -c network.c $(CFLAGS) $(CPP_FLAGS) $(LIBS)

mm.o: mm.c
	$(CC) -c mm.c $(CFLAGS) $(CPP_FLAGS) $(LIBS)

arr.o: arr.c
	$(CC) -c arr.c $(CFLAGS) $(CPP_FLAGS) $(LIBS)
