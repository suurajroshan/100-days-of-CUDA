# Compiler and flags
NVCC = nvcc
CCFLAGS = -I./include -rdc=true

# Target executable
TARGET = main

# Directories
SRC_DIR = src
OBJ_DIR = obj
INCLUDE_DIR = include

# Source and object files
SRCS = $(SRC_DIR)/main.cu $(SRC_DIR)/helper_functions.cu $(SRC_DIR)/multiplicationKernels.cu
OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))

# Default target
all: $(TARGET)

# Ensure object directory exists before compiling
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CCFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Link all object files into the final executable
$(TARGET): $(OBJS)
	$(NVCC) -rdc=true $(CCFLAGS) $^ -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR) $(TARGET)
