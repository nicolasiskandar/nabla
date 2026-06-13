#ifndef NABLA_CONTEXT_H
#define NABLA_CONTEXT_H

#include "ast.h"
#include "type.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <csetjmp>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

extern std::map<std::string, double> g_globals;
extern std::set<std::string> g_globals_immutable;
extern bool g_repl_mode;
extern bool g_bounds_err;
extern std::jmp_buf g_bounds_jmp;
extern std::map<std::string, std::string> g_var_types;
extern std::map<std::string, int> g_var_lens;

namespace llvm
{
namespace orc
{
class NablaJIT;
}
} // namespace llvm

extern std::unique_ptr<llvm::LLVMContext> g_ctx;
extern std::unique_ptr<llvm::Module> g_mod;
extern std::unique_ptr<llvm::IRBuilder<>> g_ir;
extern std::map<std::string, llvm::AllocaInst *> g_vars;
extern std::unique_ptr<llvm::orc::NablaJIT> g_jit;
extern std::map<std::string, std::unique_ptr<ProtoNode>> g_protos;
extern std::map<std::string, int> g_op_prec;

extern llvm::BasicBlock *g_break_bb;
extern llvm::BasicBlock *g_cont_bb;

struct DiState
{
    llvm::DICompileUnit *cu = nullptr;
    llvm::DIType *dbl_ty = nullptr;
    std::vector<llvm::DIScope *> scopes;
    void emit_pos(Node *n);
    llvm::DIType *dbl_ref();
};

extern DiState g_di;
extern std::unique_ptr<llvm::DIBuilder> g_dbg;

extern std::map<std::string,
                std::vector<std::pair<std::string, std::unique_ptr<TypeNode>>>>
    g_struct_types;

extern std::map<std::string, std::vector<int>> g_var_dims;

void g_prec_init();
void setup_mod();

#endif
