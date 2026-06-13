# Architecture

High-level overview of the Nabla compiler pipeline, runtime model, and execution flow.

---

## Compiler Pipeline

Nabla is a single-pass compiler: source → tokens → AST → LLVM IR → machine code through ORC JIT.

```
┌─────────┐     ┌────────┐     ┌────────┐     ┌──────────┐     ┌─────────┐
│ Source  │ --> │ Lexer  │ --> │ Parser │ --> │ Codegen  │ --> │ ORC JIT │
│  Code   │     │ (scan) │     │ (ast)  │     │(llvm ir) │     │ (exec)  │
└─────────┘     └────────┘     └────────┘     └──────────┘     └─────────┘
```

---

## Phase 1: Lexical Analysis

**Module**: `src/lexer.cpp`, `src/token.cpp`

### Responsibilities

- Convert source text into a stream of tokens
- Track line and column positions for diagnostics
- Handle nested block comments: `/* outer /* inner */ still outer */`
- Recognize indentation-sensitive syntax

### Key Structures

```cpp
struct Token {
    int kind;              // TOKEN_* constants
    std::string lexeme;    // Text representation
    int line, col;         // Position info
    double num_val;        // For numeric literals
};
```

### Token Types

| Category   | Examples                                 |
| ---------- | ---------------------------------------- |
| Literals   | `NUMBER`, `INT`, `STRING`, `BOOL`        |
| Keywords   | `let`, `mut`, `fn`, `if`, `while`, `for` |
| Operators  | `+`, `-`, `*`, `/`, `@`, `==`, `!=`      |
| Delimiters | `(`, `)`, `[`, `]`, `{`, `}`, `,`        |
| Special    | `EOF`, `NEWLINE`                         |

---

## Phase 2: Parsing

**Module**: `src/parser.cpp`

### Responsibilities

- Build an Abstract Syntax Tree (AST) from tokens
- Validate syntax and report parse errors
- Resolve operator precedence and associativity
- Record type annotations and matrix dimensions

### Parsing Strategy

Recursive-descent parser with operator precedence climbing for expressions. Precedence table in `src/codegen.cpp` (`g_op_prec`).

### AST Node Types

| Construct       | Node Class                                      |
| --------------- | ----------------------------------------------- |
| Constants       | `NumNode`, `IntNode`, `StrNode`, `BoolNode`     |
| Variables       | `SymNode`                                       |
| Operators       | `UnaryNode`, `BinaryNode`                       |
| Function calls  | `CallNode`                                      |
| Control flow    | `IfNode`, `WhileNode`, `ForNode`, `SwitchNode`  |
| Bindings        | `LetNode`, `GlobalVarNode`                      |
| Collections     | `ArrNode`, `IdxNode`, `FieldNode`, `StructNode` |
| Functions       | `ProtoNode`, `FuncNode`                         |
| Matrix multiply | `BinaryNode` with op="@"                        |
| Gradients       | `CallNode` with func="grad" or func="∇"         |

### Example: Parsing `x + y * 2`

1. Parse `x` → `SymNode("x")`
2. See `+` (precedence 20)
3. Parse RHS with precedence > 20
4. Parse `y` → `SymNode("y")`
5. See `*` (precedence 40 > 20) → continue RHS
6. Parse `2` → `NumNode(2.0)`
7. Create `BinaryNode("*", y, 2)`
8. Back to top: Create `BinaryNode("+", x, y*2)`

---

## Phase 3: Type Annotation & Dimension Tracking

**Module**: `src/parser.cpp` (during parsing), dimension globals in `src/codegen.cpp`

### Global Type Metadata

The parser populates dimension and type information:

```cpp
std::map<std::string, std::vector<int>> s_var_dims;  // Variable → shape
std::map<std::string, std::string> s_var_types;      // Variable → type string
```

Examples:

- `let A: matrix[2, 3]` → `s_var_dims["A"] = {2, 3}`, `s_var_types["A"] = "matrix[2, 3]"`
- `let x = 3.14` → `s_var_types["x"] = "float"`

This enables compile-time validation:

- Matrix multiplication `@` checks dimension compatibility
- Transpose `'` swaps dimensions
- `grad` and `jacobian` validate function signatures

---

## Phase 4: Code Generation

**Module**: `src/codegen.cpp`

### Responsibilities

- Lower AST to LLVM IR
- Manage symbol tables and variable allocations
- Emit control flow graphs (basic blocks)
- Implement automatic type promotion and matrix operations
- Track matrix shape information for validation

### LLVM IR Basics

Nabla targets LLVM 22. Each Nabla value maps to an LLVM value:

| Nabla Type     | LLVM Type              | Storage           |
| -------------- | ---------------------- | ----------------- |
| `float`        | `i64` (as double bits) | Stack or register |
| `int`          | `i64`                  | Stack or register |
| `bool`         | `i64` (0/1)            | Stack or register |
| `string`       | `i8*`                  | Pointer to heap   |
| `[float]`      | `double*`              | Pointer to heap   |
| `matrix[N, M]` | `double*`              | Pointer to heap   |
| `struct`       | LLVM struct            | Stack (by-value)  |

### Key Functions

#### `Value* compile(Node* node)`

Recursively compile an AST node to LLVM IR. Returns the LLVM `Value*` representing the result.

```cpp
Value* NumNode::compile() {
    return ConstantFP::get(TheContext, APFloat(num_val));
}

Value* BinaryNode::compile() {
    Value* lval = lhs->compile();
    Value* rval = rhs->compile();
    if (op == "+") return Builder.CreateFAdd(lval, rval);
    // ... handle other operators
}
```

#### Symbol Tables

| Global                | Purpose                                             | Type                        |
| --------------------- | --------------------------------------------------- | --------------------------- |
| `g_vars`              | Stack allocations (`alloca`)                        | `map<string, Value*>`       |
| `g_var_types`         | Inferred type names (e.g., "float", "matrix[2, 3]") | `map<string, string>`       |
| `g_var_dims`          | Matrix dimensions                                   | `map<string, vector<int>>`  |
| `g_var_lens`          | Array lengths                                       | `map<string, int>`          |
| `g_globals`           | REPL-level bindings                                 | `map<string, GenericValue>` |
| `g_globals_immutable` | Names bound with `let` (vs. `mut`)                  | `set<string>`               |

#### Control Flow

Each construct (if, while, for) emits LLVM basic blocks:

```cpp
// if (cond) then a else b
BasicBlock* then_bb = BasicBlock::Create(TheContext, "then", TheFunction);
BasicBlock* else_bb = BasicBlock::Create(TheContext, "else", TheFunction);
BasicBlock* merge_bb = BasicBlock::Create(TheContext, "merge", TheFunction);

Value* cond_val = cond->compile();
Builder.CreateCondBr(cond_val, then_bb, else_bb);

// Emit "then" block
Builder.SetInsertPoint(then_bb);
Value* then_val = a->compile();
Builder.CreateBr(merge_bb);

// Emit "else" block
Builder.SetInsertPoint(else_bb);
Value* else_val = b->compile();
Builder.CreateBr(merge_bb);

// Phi node to merge results
Builder.SetInsertPoint(merge_bb);
PHINode* result = Builder.CreatePHI(Type, 2);
result->addIncoming(then_val, then_bb);
result->addIncoming(else_val, else_bb);
```

#### Matrix Operations

- **Multiply `@`**: Calls runtime helper `__nabla_matmul(A_ptr, A_rows, A_cols, B_ptr, B_rows, B_cols)`
- **Transpose `'`**: Creates new allocation and copies transposed elements
- **Dot product**: Element-wise multiply, then sum

#### Automatic Differentiation

- **`grad(f, x)`**: Compiles using symbolic differentiation or runtime autodiff
- **Type checking**: Validates that `f` returns a scalar and `x` is a matrix
- **Dimension tracking**: Ensures output gradient matches input shape

### Runtime Helpers

Matrix operations delegate to C runtime functions:

| Function         | Signature                                        | Purpose            |
| ---------------- | ------------------------------------------------ | ------------------ |
| `__nabla_matmul` | `double* (double*, i32, i32, double*, i32, i32)` | Matrix multiply    |
| `__nabla_solve`  | `double* (double*, i32, double*)`                | Linear system      |
| `__nabla_inv`    | `double* (double*, i32)`                         | Matrix inverse     |
| `__nabla_det`    | `double (double*, i32)`                          | Determinant        |
| `__nabla_eig`    | `double* (double*, i32)`                         | Eigenvalues        |
| `__nabla_mean`   | `double (double*, i32)`                          | Mean               |
| `__nabla_std`    | `double (double*, i32)`                          | Standard deviation |
| `__nabla_norm`   | `double (double*, i32)`                          | Euclidean norm     |

---

## Phase 5: JIT Compilation & Execution

**Module**: `src/jit.cpp`, `src/main.cpp`

### LLVM ORC JIT

Nabla uses LLVM 22's ORC (On-Request Compilation) JIT engine:

```cpp
LLJIT jit;
jit.addIRModule(std::move(module));
jit.lookup("main");  // Compile and execute
```

### Execution Modes

#### File Mode

1. Parse entire `.nbl` file
2. Compile all statements into a `main()` function
3. JIT and execute `main`
4. Output results to stderr

#### REPL Mode

1. Read one line from stdin
2. Compile as `__expr_N()` function
3. JIT and execute
4. For `let`/`mut` bindings: print inferred type and value
5. For expressions: print result value
6. Return to step 1

### Global State Persistence

In REPL, bindings persist across lines using `g_globals`:

```
> let x = 5.0
x: float = 5.0

> let y = x + 3.0
y: float = 8.0

> x + y
13.000000
```

The `g_globals` map stores `GenericValue`s that survive across JIT compilations.

---

## Type System Details

### Type Categories

Nabla groups types into categories for validation:

| Category  | Types                  | Pointer-like |
| --------- | ---------------------- | ------------ |
| Scalar    | `float`, `int`, `bool` | No           |
| Text      | `string`, `str`        | Yes          |
| Sequence  | `[type]`               | Yes          |
| Matrix    | `matrix[N, M]`         | Yes          |
| Aggregate | `struct`               | No           |

**Helper**: `is_pointer_like_type(const string& type_name)` returns true for string, array, and matrix types.

### Type Inference

- **Literals**: `3.14` → float, `42` → int, `"text"` → string
- **Operations**: Result type follows common type (float if any operand is float)
- **Annotations**: Explicit types override inference
- **Mutable variables**: Cannot change type category after initialization

---

## Error Handling

### Compile-Time Errors

Caught during parsing and codegen:

- Type mismatches (e.g., `int` assigned to `float` without promotion)
- Dimension mismatch in matrix operations (e.g., `matrix[2,3] @ matrix[3,5]` OK, but `matrix[2,3] @ matrix[2,2]` fails)
- Undefined variables or functions
- Type-category violations (e.g., assigning a scalar to a `mut` matrix)

### Runtime Errors

Possible at JIT execution:

- Division by zero
- Array out-of-bounds
- FFI symbol not found

---

## Directory Structure

```
nabla/
├── CMakeLists.txt        # Build configuration
├── include/              # Public headers
│   ├── lexer.h
│   ├── parser.h
│   ├── codegen.h
│   ├── jit.h
│   ├── ast.h
│   ├── type.h
│   └── ... (others)
├── src/                  # Implementation
│   ├── lexer.cpp
│   ├── parser.cpp
│   ├── codegen.cpp
│   ├── jit.cpp
│   ├── main.cpp          # REPL and file mode
│   ├── runtime.cpp       # Matrix/math helpers
│   └── ... (others)
├── docs/                 # Documentation
├── examples/             # Example programs
└── build/                # Build artifacts (generated)
```

---

## Memory Model

### Stack Allocation

- Scalar values (float, int, bool) stored as immediates or in registers
- Struct values (all fields inline) stack-allocated
- All local variables → `alloca` + store/load

### Heap Allocation

- Strings, arrays, matrices → pointer-to-first-element
- Allocated by runtime helpers; lifetime tied to JIT session
- No garbage collection for now, still not sure whether to manually free global vars or use GC; memory frees only when session ends

### Scope

Variables live in `let...in` expressions. Upon exit, local allocas become inaccessible (still on stack until frame ends).

Example:

```nbl
let x = 5.0 in
    let y = x + 1.0 in
        print(y)    # x and y in scope
    # y out of scope here (but x still in scope)
# x out of scope here
```

---

## Limits & Implementation Notes

| Aspect            | Limit                 | Reason                     |
| ----------------- | --------------------- | -------------------------- |
| Matrix dimensions | Compile-time constant | Enables early validation   |
| Recursion         | Stack depth           | JIT runtime stack          |
| Array size        | Heap memory           | Allocation at runtime      |
| String length     | Memory available      | No length limit in type    |
| REPL history      | Memory available      | Globals persist in session |

### Known Limitations

- No first-class closures (functions capture by reference, not value)
- No list comprehension or higher-order map/filter
- Structs immutable after creation
- No generics or templates
- Strings indexed by element count, not by byte offset (multi-byte UTF-8 may behave unexpectedly)
