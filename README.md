# XTL

A from-scratch N-dimensional tensor library in C with Python bindings, capable of running GPT-2 inference.

## Features

- Row-major tensors with explicit stride-based indexing
- Broadcasting for all elementwise ops
- Batched matrix multiplication with broadcasting over batch dims
- Shape ops: reshape, transpose, contiguous copy
- Reductions: sum, max, mean, var (over any axis)
- Activations: gelu, gelu_apprx, sigmoid, tanh, exp, sqrt, log, neg
- Scalar ops: add_scalar, mul_scalar, div_scalar
- Composed ops: softmax, layer_norm, gather (embedding lookup)
- Python bindings via CFFI — reads the C header directly, no code generation step

## Build

Requires `gcc` and `libm`.

```bash
make          # builds python/libxtl.so
make clean    # removes the shared library
```

## Python

Uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
cd python
uv sync
uv run pytest tests/   # 103 tests
```

## GPT-2 Chat

Download the weights (523 MB) then start the REPL:

```bash
cd python
curl -L https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors \
     -o model.safetensors
uv run python gpt2.py
```

```
Loading GPT-2 weights from model.safetensors ...
Loaded in 1.5s
Type your prompt and press Enter. Ctrl+C or Ctrl+D to quit.

You: Who was Alan Turing
GPT-2: Alan Turing was a British mathematician and computer scientist...
```

GPT-2 is a text completer — the Q:/A: format used by the REPL steers it into answer mode.
Expect ~5 seconds per token on CPU (unoptimized triple-loop matmul).

## Project layout

```
lib/
  tensor/
    tensor.h              # public API
    cpu/
      tensor_alloc.c      # init_tensor, free_tensor
      tensor_create.c     # ones, zeros, rand, from_data
      tensor_fill.c       # fill_tensor, fill_tensor_random
      tensor_print.c      # print_tensor
      tensor_ops.c        # elementwise, matmul, reductions, softmax, layer_norm, gather
      tensor_shapes.c     # reshape, transpose, contiguous
      utils.c             # BroadcastCtx helpers
python/
  xtl.py                  # CFFI wrapper
  gpt2.py                 # GPT-2 forward pass + chat REPL
  tests/                  # pytest test suite
```
