# ∇ Nabla

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LLVM](https://img.shields.io/badge/LLVM-22-262D3A?logo=llvm)](https://llvm.org)
[![Build](https://img.shields.io/badge/build-cmake-blue)](CMakeLists.txt)

A compiled programming language for numerical computing, where differentiation
is part of the syntax, not the standard library.

Nabla treats matrices, arrays, and numerical differentiation as first-class
language constructs. Write `∇f(x)` and the compiler generates a gradient,
computed via central finite differences, with native matrix types, LU-based
linear solvers, and an interactive LLVM JIT.

## Why Nabla

Most numerical code is written in a general-purpose language and made
numerical via libraries (NumPy, Eigen, BLAS). Nabla inverts this: the language
itself understands what a matrix is, what `A @ B` means dimensionally, and
what `∇f(x)` should produce.

## Features

- **Native matrix and array types** — `matrix[m, n]`, with shape-checked
  operations (`@` for matmul, `'` for transpose)
- **∇ as syntax** — `∇f(x)` computes the gradient of `f` at `x` via central
  finite differences; `jacobian(f, x)` for vector-valued functions
- **Linear algebra runtime** — `solve`, `inv`, `det` (LU decomposition with
  partial pivoting), `eig` (cyclic Jacobi, symmetric matrices)
- **Statistics & vectors** — `mean`, `std`, `cov`, `corr`, `norm`, `linspace`,
  `range`
- **LLVM 22 ORC JIT** — compiles to native machine code; REPL and file-mode
  execution
- **Structs, control flow, custom operators** — `if/elif/else`, `while`,
  `for`, `switch`, user-defined binary/unary operators with precedence

## Quick Start

```bash
cmake -B build
cmake --build build
./build/nabla examples/00_hello.nbl
```

Launch the REPL with `./build/nabla`.

## Quick Example

```nbl
fn f(x: matrix[3, 1])
    x[0] ** 2 + x[1] ** 2 + x[2] ** 2

∇f([3.0, 4.0, 0.0])
```

```nbl
let A: matrix[2, 2] = [4.0, 7.0, 2.0, 6.0] in
let b: matrix[2, 1] = [1.0, 2.0] in
    solve(A, b)
```

## Learn More

| Document                                   | Best For                 |
| ------------------------------------------ | ------------------------ |
| [Getting Started](docs/GETTING_STARTED.md) | Install, build, run      |
| [Language Reference](docs/LANGUAGE.md)     | Syntax, types, operators |
| [Builtins Reference](docs/BUILTINS.md)     | Runtime functions        |
| [Architecture](docs/ARCHITECTURE.md)       | Compiler pipeline        |
| [Examples](examples/README.md)             | Runnable learning path   |

## Common Use Cases

- Numerical computing with static shapes and predictable performance
- Data analysis with built-in statistics and vector utilities
- Scientific prototyping with a REPL-first workflow

## License

MIT
