# Language Reference

Concise guide to Nabla syntax, types, and semantics.

---

## Comments

Nabla supports three comment styles:

```nbl
# Single-line comment
// Another single-line style
/* Multi-line comment */
/* Nested comments /* work fine */ also */
```

---

## Type System

### Primitive Types

| Type    | Syntax           | Literal Example       | Notes                   |
| ------- | ---------------- | --------------------- | ----------------------- |
| Float   | `float`          | `3.14`, `1.0e-2`      | 64-bit IEEE 754         |
| Integer | `int`            | `42`, `-7`            | 64-bit signed           |
| Boolean | `bool`           | `true`, `false`       | Truth values            |
| String  | `string` / `str` | `"hello"`, `"line\n"` | UTF-8, escape sequences |
| Void    | `void`           | —                     | Empty return type       |

**Literal Type Inference**:

- Numbers without decimal point → `int`
- Numbers with decimal point → `float`
- Text in quotes → `string`

### Composite Types

| Type   | Syntax                | Example             | Semantics               |
| ------ | --------------------- | ------------------- | ----------------------- |
| Array  | `[type]`              | `[1.0, 2.0]`        | Homogeneous sequences   |
| Matrix | `matrix[N, M]`        | `matrix[3, 2]`      | Static-size 2D arrays   |
| Struct | `struct Name { ... }` | See Structs section | User-defined aggregates |

### Type Promotion

Mixed `int`/`float` expressions automatically promote `int` to `float`:

```nbl
print(2 + 3.0)       # 5.0 (int + float → float)
print(10.0 / 3)      # 3.333333 (10.0 / int → float)
print(2 ** 3)        # 8 (int ** int → int)
print(2.0 ** 3)      # 8.0 (float ** int → float)
```

### Type Annotations

Explicitly specify types to override inference:

```nbl
let x: int = 42 in x + 1           # OK
let x: int = 42.5 in x             # Error: cannot assign float to int
let A: matrix[2, 3] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] in A[0, 0]
```

---

## Expressions

### Arithmetic Operators

| Operator | Types      | Precedence | Example        |
| -------- | ---------- | ---------- | -------------- |
| `+`      | float, int | 20         | `3 + 4` → 7    |
| `-`      | float, int | 20         | `10 - 3` → 7   |
| `*`      | float, int | 40         | `3 * 4` → 12   |
| `/`      | float, int | 40         | `10 / 2` → 5.0 |
| `**`     | float, int | 60         | `2 ** 3` → 8.0 |

**Associativity**: Left-to-right for `+`, `-`, `*`, `/`; right-to-left for `**`.

```nbl
2 + 3 * 4        # → 14 (multiplication first)
10 - 4 - 2       # → 4 (left-to-right)
2 ** 3 ** 2      # → 512 (right-to-left: 2^(3^2))
```

### Comparison Operators

| Operator | Returns               | Example      |
| -------- | --------------------- | ------------ |
| `==`     | 1 (true) or 0 (false) | `3 == 3` → 1 |
| `!=`     | 1 or 0                | `3 != 4` → 1 |
| `<`      | 1 or 0                | `3 < 5` → 1  |
| `<=`     | 1 or 0                | `5 <= 5` → 1 |
| `>`      | 1 or 0                | `6 > 5` → 1  |
| `>=`     | 1 or 0                | `5 >= 3` → 1 |

Comparisons on different types auto-promote:

```nbl
3 < 5.0          # → 1 (int promoted to float)
"abc" == "abc"   # → 1 (string comparison)
```

### Logical Operators

| Operator | Semantics                   | Example            |
| -------- | --------------------------- | ------------------ |
| `&&`     | Logical AND (short-circuit) | `1 && 0` → 0       |
| `\|\|`   | Logical OR (short-circuit)  | `0 \|\| 1` → 1     |
| `!`      | Logical NOT                 | `!1` → 0, `!0` → 1 |

Non-zero values are truthy:

```nbl
if 42 then print("truthy")                   # Prints (42 is non-zero)
if 0 then print("skip") else print("falsy")  # Prints "falsy"
```

### Matrix Operations

| Operator    | Syntax      | Example                          | Requires                         |
| ----------- | ----------- | -------------------------------- | -------------------------------- |
| Multiply    | `A @ B`     | `matrix[2,3] @ matrix[3,4]`      | Dimension match known at compile |
| Transpose   | `A'`        | `matrix[2, 3]'` → `matrix[3, 2]` | Static dimensions                |
| Dot Product | `dot(u, v)` | `dot([1, 2], [3, 4])` → 11       | Vector dimensions                |

```nbl
let A: matrix[2, 2] = [1.0, 2.0, 3.0, 4.0] in
let B: matrix[2, 3] = [1.0, 0.0, 2.0, 3.0, 1.0, 0.0] in
    print(A @ B)  # 2×3 result
```

### Indexing

Access array, matrix, and string elements with zero-based indexing:

```nbl
let a = [10.0, 20.0, 30.0] in
    print(a[0])      # 10.0

let m: matrix[2, 2] = [1.0, 2.0, 3.0, 4.0] in
    print(m[0, 0])   # First element (row-major): 1.0

let s = "hello" in
    print(s[0])      # h
```

**Index Mutation** (requires `mut`):

```nbl
let a = [1.0, 2.0, 3.0]
a[1] = 99.0
print(a)
```

### Conditional Expression

The `if`-`then`-`else` construct is an expression, not a statement:

```nbl
let x = 1.0 in
let result = if x > 0.0 then "positive" else "non-positive" in
    print(result)

let a = 5.0 in
let b = 3.0 in
let max_val = if a > b then a else b in
    print(max_val)
```

Chain conditions with `elif`:

```nbl
if x < 0.0 then print("negative")
elif x == 0.0 then print("zero")
else print("positive")
```

### String Operations

**Concatenation**:

```nbl
let greeting = "Hello, " + "World!" in print(greeting)
```

**Indexing**:

```nbl
let s = "Nabla" in print(s[0])   # N
```

**Length**:

```nbl
let s = "hello" in print(len(s))  # 5.0
```

---

## Variables

### Immutable Binding: `let`

Binds a value in a scope. Immutable.

```nbl
let x = 3.14 in print(x + 1.0)

let x = 10
x = 20 # Error: cannot assign to immutable variable
```

**Type Annotation**:

```nbl
let x: float = 42 in print(x)    # x is explicitly float
let A: matrix[3, 3] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] in
    print(A[1, 1])
```

**Nested Bindings**:

```nbl
let a = 1.0 in
let b = 2.0 in
    print(a + b)   # → 3.0
```

### Mutable Binding: `mut`

Like `let`, but allows reassignment:

```nbl
mut counter = 0
counter = 1
counter = 2
print(counter)   # 2
```

**Type Stability**: Mutable bindings cannot change type categories:

```nbl
mut x: float = 5.0
x = 3.14     # OK: float to float
print(x)

mut m: matrix[2, 2] = [1.0, 2.0, 3.0, 4.0]
m = [5.0]         # Error: cannot assign non-matrix value to matrix variable 'm'
print(m)
```

### Multiple Bindings

Bind multiple variables in one expression:

```nbl
let x = 1.0, y = 2.0, z = 3.0 in
    print(x + y + z)   # → 6.0
```
