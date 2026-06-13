# ∇ Nabla

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LLVM](https://img.shields.io/badge/LLVM-22-262D3A?logo=llvm)](https://llvm.org)
[![Build](https://img.shields.io/badge/build-cmake-blue)](CMakeLists.txt)

Nabla is a compiled language for numerical computing with native arrays, matrices, automatic differentiation, and an interactive JIT workflow.

It is built for fast iteration: write math-heavy code, run it in a REPL or from a file, and execute directly through LLVM.

## What Nabla Gives You

- Native matrix types in the language, not as a helper library
- Automatic differentiation with `grad` and `∇`
- Linear algebra runtime support for `solve`, `inv`, `det`, and `eig`
- LLVM 22 ORC JIT execution with native machine code output
- Persistent REPL state for interactive numerical work

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

| Document | Best For |
| --- | --- |
| [Getting Started](docs/GETTING_STARTED.md) | Install, build, run |
| [Language Reference](docs/LANGUAGE.md) | Syntax, types, operators |
| [Builtins Reference](docs/BUILTINS.md) | Runtime functions |
| [Architecture](docs/ARCHITECTURE.md) | Compiler pipeline |
| [Examples](examples/README.md) | Runnable learning path |

## Common Use Cases

- Numerical computing with static shapes and predictable performance
- Optimization loops powered by autodiff
- Data analysis with built-in statistics and vector utilities
- Scientific prototyping with a REPL-first workflow

## License

MIT
