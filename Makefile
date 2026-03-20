CC     = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -fopenmp -lm

# matmul_kernel.c is compiled WITHOUT -fopenmp so gcc can freely
# auto-vectorise the serial tiled kernel with AVX2 FMA.
KERNEL_OBJ = build/matmul_kernel.o

LIB_SRC = lib/tensor/cpu/tensor_alloc.c \
           lib/tensor/cpu/tensor_fill.c \
           lib/tensor/cpu/tensor_print.c \
           lib/tensor/cpu/tensor_create.c \
           lib/tensor/cpu/tensor_ops.c \
           lib/tensor/cpu/tensor_shapes.c \
           lib/tensor/cpu/utils.c

$(KERNEL_OBJ): lib/tensor/cpu/matmul_kernel.c | build
	$(CC) -Wall -Wextra -O3 -march=native -c $< -o $@

libxtl.so: $(LIB_SRC) $(KERNEL_OBJ)
	$(CC) $(CFLAGS) -shared -fPIC -o python/libxtl.so $(LIB_SRC) $(KERNEL_OBJ)

build:
	mkdir -p build

clean:
	rm -f python/libxtl.so
	rm -rf build
