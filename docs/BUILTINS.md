# Builtins Reference

Runtime functions for I/O, math, statistics, linear algebra, and autodiff.

All builtins are available without `extern` declarations. Output is written to `stderr`.

---

## I/O Functions

### `print(x)`

Print a value followed by a newline.

| Type     | Output Format                           |
| -------- | --------------------------------------- |
| `float`  | Decimal notation (e.g., `3.141593`)     |
| `int`    | Decimal (e.g., `42`)                    |
| `bool`   | `true` or `false`                       |
| `string` | Raw text                                |
| `[type]` | Space-separated elements                |
| `matrix` | Row-major with line breaks and brackets |

```nbl
print(3.14)             # 3.141593
print(42)               # 42
print(true)             # true
print([1.0, 2.0, 3.0])  # [1.000000, 2.000000, 3.000000]
```

### `printd(x)`

Print a number without trailing newline. Useful when building a line manually.

```nbl
printd(1.0) + printd(" ") + printd(2.0) + printd("\n")
# Output: 1.000000 2.000000
```

### `putchard(n)`

Print one ASCII character. Accepts a number corresponding to ASCII code.

```nbl
putchard(65)   # Prints 'A' (ASCII 65)
putchard(72)   # Prints 'H' (ASCII 72)
```

### `input()`

Read one line from stdin and return it as a string.

```nbl
let name = input() in
    print("Hello, " + name)
```

### `type(x)`

Print the inferred type of `x` for debugging and introspection.

```nbl
let x = 3.14 in
    type(x)        # Output: float

let A: matrix[2, 3] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] in
    type(A)        # Output: matrix[2, 3]
```

### `len(x)`

Get the length or size of a sequence.

| Type           | Returns | Notes                  |
| -------------- | ------- | ---------------------- |
| `string`       | `float` | Number of characters   |
| `[type]`       | `float` | Number of elements     |
| `matrix[N, M]` | `float` | Total elements (N × M) |

```nbl
len("hello")                             # 5.0
len([1.0, 2.0, 3.0])                     # 3.0
let A: matrix[2, 3] = [...] in len(A)    # 6.0
```

---

## Math Functions

### `sqrt(x)`

Compute the square root of `x`.

```nbl
sqrt(4.0)       # 2.0
sqrt(2.0)       # 1.414214
```

### `sin(x)`, `cos(x)`, `tan(x)`

Trigonometric functions from the C standard library.

```nbl
sin(0.0)        # 0.0
cos(0)          # 1.0
tan(1)          # 1.557408
```

---

## Array & Matrix Constructors

### `zeros(n)` / `zeros(r, c)`

Create a zero-filled vector or matrix.

```nbl
zeros(3)        # matrix[1, 3]: [0.0, 0.0, 0.0]
zeros(2, 3)     # matrix[2, 3]: zero matrix
```

### `ones(n)` / `ones(r, c)`

Create a matrix filled with ones.

```nbl
ones(4)         # matrix[1, 4]: [1.0, 1.0, 1.0, 1.0]
ones(2, 2)      # matrix[2, 2]: [[1.0, 1.0], [1.0, 1.0]]
```

### `eye(n)`

Create an identity matrix of size `n × n`.

```nbl
eye(3)          # matrix[3, 3]: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
```

### `linspace(a, b, n)`

Generate `n` evenly spaced values from `a` to `b` (inclusive).

```nbl
linspace(0.0, 1.0, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
linspace(1.0, 3.0, 3)  # [1.0, 2.0, 3.0]
```

### `range(a, b)`

Generate integers from `a` to `b` (inclusive, both as `int`).

```nbl
range(1, 5)    # [1, 2, 3, 4, 5]
range(0, 10)   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

---

## Statistical Functions

### `mean(v)` / `avr(v)`

Compute the arithmetic mean of a vector.

```nbl
mean([1.0, 2.0, 3.0, 4.0])  # 2.5
avr([2.0, 4.0, 6.0])        # 4.0
```

### `std(v)`

Compute the sample standard deviation.

```nbl
std([1.0, 2.0, 3.0])  # 1.0
```

### `norm(v)`

Compute the Euclidean norm (L₂ norm) of a vector.

```nbl
norm([3.0, 4.0])  # 5.0 (3² + 4² = 25, √25 = 5)
norm([1.0, 1.0, 1.0])  # 1.732051 (√3)
```

### `cov(x, y)`

Compute the sample covariance between vectors `x` and `y`.

```nbl
let x = [1.0, 2.0, 3.0] in
let y = [2.0, 4.0, 6.0] in
    cov(x, y)   # 2.0
```

### `corr(x, y)`

Compute the Pearson correlation coefficient.

```nbl
let x = [1.0, 2.0, 3.0] in
let y = [2.0, 4.0, 5.0] in
    corr(x, y)  # 0.981981 (strong positive correlation)
```

---

## Linear Algebra Functions

### `det(A)`

Compute the determinant of a square matrix.

```nbl
let A: matrix[2, 2] = [4.0, 7.0, 2.0, 6.0] in
    det(A)      # 10.0
```

### `solve(A, b)`

Solve the linear system `Ax = b` for `x`.

| Input             | Output         | Notes           |
| ----------------- | -------------- | --------------- |
| `A: matrix[n, n]` | `matrix[n, 1]` | Solution vector |
| `b: matrix[n, 1]` | —              | Right-hand side |

```nbl
let A: matrix[2, 2] = [4.0, 7.0, 2.0, 6.0] in
let b: matrix[2, 1] = [1.0, 2.0] in
    solve(A, b)
    # Returns the vector x such that Ax = b
```

### `inv(A)`

Compute the matrix inverse of `A`.

```nbl
let A: matrix[2, 2] = [4.0, 7.0, 2.0, 6.0] in
let Ainv = inv(A) in
    print(Ainv)
```

### `eig(A)`

Compute the eigenvalues of a symmetric matrix `A`.

Implementation note: Nabla uses iterative Jacobi rotations on a working copy of
`A` until the off-diagonal terms converge toward zero, then reads the diagonal
entries as the eigenvalues. The result is returned as a `matrix[1, n]` row
matrix.

```nbl
let A: matrix[2, 2] = [1.0, 0.5, 0.5, 2.0] in
    eig(A)  # Returns eigenvalues as a 1xn row matrix
```

---

## Automatic Differentiation

### `grad(f, x)`

Compute the gradient of a function `f` at point `x`.

| Argument    | Type                                    |
| ----------- | --------------------------------------- |
| `f`         | Function mapping `matrix[1, n] → float` |
| `x`         | Input point `matrix[1, n]`              |
| **Returns** | Gradient vector `matrix[1, n]`          |

```nbl
fn f(x: matrix[1, 2])
    x[0, 0] * x[0, 0] + 2.0 * x[0, 1] * x[0, 1]

let x: matrix[1, 2] = [1.0, 2.0] in
let g: matrix[1, 2] = grad(f, x) in
    print(g)  # [[2.0, 8.0]] (∂f/∂x₀ = 2x₀, ∂f/∂x₁ = 4x₁)
```

### `∇f(x)` (Unicode Nabla)

Shorthand for `grad(f, x)`.

```nbl
fn f(x: matrix[1, 1])
    x[0, 0] ** 2

let x: matrix[1, 1] = [3.0] in
let g: matrix[1, 1] = ∇f(x) in
    print(g)  # [[6.0]]
```

### `jacobian(f, x)`

Compute the Jacobian matrix of a vector-valued function.

```nbl
fn f(x: matrix[1, 2])
    -> matrix[1, 2]
    [x[0, 0] * x[0, 1], x[0, 0] * x[0, 0]]

let x: matrix[1, 2] = [2.0, 3.0] in
let J: matrix[2, 2] = jacobian(f, x) in
    print(J)  # [[3.0, 2.0], [4.0, 0.0]]
```

---

## Full Function Signatures

| Function            | Signature                             | Returns                           |
| ------------------- | ------------------------------------- | --------------------------------- |
| `sqrt`              | `(x: float)`                          | `float`                           |
| `sin`, `cos`, `tan` | `(x: float)`                          | `float`                           |
| `eye`               | `(n: int)`                            | `matrix[n, n]`                    |
| `zeros`             | `(n: int)` \| `(r: int, c: int)`      | `matrix[1, n]` \| `matrix[r, c]`  |
| `ones`              | `(n: int)` \| `(r: int, c: int)`      | `matrix[1, n]` \| `matrix[r, c]`  |
| `linspace`          | `(a: float, b: float, n: int)`        | `matrix[1, n]`                    |
| `range`             | `(a: int, b: int)`                    | `matrix[1, b-a+1]`                |
| `mean` / `avr`      | `(v: [float])` \| `(v: matrix[...])`  | `float`                           |
| `std`               | `(v: [float])` \| `(v: matrix[...])`  | `float`                           |
| `norm`              | `(v: [float])` \| `(v: matrix[...])`  | `float`                           |
| `cov`               | `(x: [...], y: [...])`                | `float`                           |
| `corr`              | `(x: [...], y: [...])`                | `float`                           |
| `det`               | `(A: matrix[n, n])`                   | `float`                           |
| `solve`             | `(A: matrix[n, n], b: matrix[n, 1])`  | `matrix[n, 1]`                    |
| `inv`               | `(A: matrix[n, n])`                   | `matrix[n, n]`                    |
| `eig`               | `(A: matrix[n, n])`                   | `matrix[1, n]`                    |
| `grad`              | `(f: fn, x: matrix[...])`             | `matrix[same shape as x]`         |
| `∇`                 | `(f: fn, x: matrix[...])`             | `matrix[same shape as x]`         |
| `jacobian`          | `(f: fn, x: matrix[...])`             | `matrix[output_size, input_size]` |
| `print`             | `(x: any)`                            | `void`                            |
| `printd`            | `(x: float \| int)`                   | `void`                            |
| `putchard`          | `(n: int)`                            | `void`                            |
| `input`             | `()`                                  | `string`                          |
| `type`              | `(x: any)`                            | `void`                            |
| `len`               | `(x: string \| [...] \| matrix[...])` | `float`                           |

---

## Common Patterns

### Normalizing a Vector

```nbl
let v = [3.0, 4.0] in
let n = norm(v) in
    print([v[0] / n, v[1] / n])  # [0.6, 0.8]
```

### Computing a Correlation Matrix

```nbl
let x = [1.0, 2.0, 3.0, 4.0, 5.0] in
let y = [2.0, 4.0, 5.0, 4.0, 6.0] in
    print(corr(x, y))
```

### Optimization with Gradient Descent

```nbl
fn loss(x: matrix[1, 2])
    (x[0, 0] - 5.0) * (x[0, 0] - 5.0) + (x[0, 1] - 3.0) * (x[0, 1] - 3.0)

let lr = 0.01 in
let x0: matrix[1, 2] = [0.0, 0.0] in
let g0 = ∇loss(x0) in
let x1 = x0[0, 0] - lr * g0[0, 0] in
let y1 = x0[0, 1] - lr * g0[0, 1] in
let x1m: matrix[1, 2] = [x1, y1] in
    print(x1m) # [0.100000, 0.060000]
```
