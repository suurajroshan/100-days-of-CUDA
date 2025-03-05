# NVCC = usr/bin/nvcc
INCLUDE = -I./include

SRC = src/main.cu src/naiveMatMulGPU.cu
OBJ = main kernel
EXE = main

all: $(EXE)

$(EXE): $(SRC)
	nvcc $(INCLUDE) $(SRC) -o $(EXE)

clean:
	rm -f $(EXE)