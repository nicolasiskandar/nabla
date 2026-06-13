# Getting Started with Nabla

Build Nabla once, then run programs or use the REPL for interactive numerics.

## Prerequisites

| Requirement  | Minimum | Notes                        |
| ------------ | ------- | ---------------------------- |
| LLVM         | 22+     | Development headers required |
| CMake        | 3.16+   | Build system generator       |
| C++ compiler | C++20   | Clang 16+, GCC 12+, or newer |

## Install Dependencies

Linux:

```bash
sudo apt-get install llvm-22-dev cmake build-essential
sudo dnf install llvm-devel cmake gcc-c++
```

macOS:

```bash
brew install llvm@22 cmake
```

## Build

```bash
cmake -B build
cmake --build build
```

For a release build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

The compiler binary is `build/nabla`.

## Verify

```bash
./build/nabla examples/00_hello.nbl
```

Expected output:

```text
Hello, Nabla!
```

## Run Your First Program

Create `my_first_program.nbl`:

```nbl
let A: matrix[2, 2] = [1.0, 2.0, 3.0, 4.0] in
let B: matrix[2, 2] = [5.0, 6.0, 7.0, 8.0] in
    print(A @ B)
```

Run it:

```bash
./build/nabla my_first_program.nbl
```

Expected output:

```text
[[19.000000, 22.000000],
 [43.000000, 50.000000]]
```

## REPL

```bash
./build/nabla
```

Example session:

```nbl
let x = [1.0, 2.0, 3.0]
mean(x)
let y = x + x
sqrt(25.0)
```

The REPL keeps bindings alive across lines, which makes it useful for incremental experiments and quick checks.

## What to Read Next

1. [Language Reference](LANGUAGE.md)
2. [Builtins Reference](BUILTINS.md)
3. [Examples](../examples/README.md)
4. [Architecture](ARCHITECTURE.md)

## Troubleshooting

| Issue                    | Fix                                       |
| ------------------------ | ----------------------------------------- |
| CMake missing            | Install `cmake` from your package manager |
| LLVM 22 missing          | Install LLVM 22 development packages      |
| Compiler errors on build | Confirm your compiler supports C++20      |
