cmake_minimum_required(VERSION 3.20)
project(nabla
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

include_directories(include)

set(SOURCES
    src/token.cpp
    src/lexer.cpp
    src/diag.cpp
    src/ast.cpp
    src/type.cpp
    src/context.cpp
    src/parser.cpp
    src/codegen.cpp
    src/builtin.cpp
    src/runtime.cpp
    src/main.cpp
)

add_executable(nabla ${SOURCES})

llvm_config(nabla
    core
    orcjit
    native
    nativecodegen
    executionengine
    passes
    support
)

target_include_directories(nabla PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_options(nabla PRIVATE -rdynamic)
target_link_libraries(nabla PRIVATE m)
