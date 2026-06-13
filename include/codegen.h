#ifndef NABLA_CODEGEN_H
#define NABLA_CODEGEN_H

#include "type.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace llvm
{
class AllocaInst;
class Function;
class Type;
} // namespace llvm

class TypeNode;

llvm::Type *llvm_type(const TypeNode *t);
llvm::Function *find_func(const std::string &name);
llvm::AllocaInst *entry_slot(llvm::Function *f, llvm::StringRef name,
                             llvm::Type *ty);

#endif
