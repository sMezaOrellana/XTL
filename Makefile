CC = gcc
CFLAGS = -Wall -Wextra -lm

LIB_SRC = lib/tensor/cpu/tensor_alloc.c \
           lib/tensor/cpu/tensor_fill.c \
           lib/tensor/cpu/tensor_print.c \
           lib/tensor/cpu/tensor_create.c \
           lib/tensor/cpu/tensor_ops.c \
           lib/tensor/cpu/tensor_shapes.c \
           lib/tensor/cpu/utils.c

libxtl.so: $(LIB_SRC)
	$(CC) $(CFLAGS) -shared -fPIC -o python/libxtl.so $(LIB_SRC)

clean:
	rm -f python/libxtl.so
