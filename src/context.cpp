#include "context.h"
#include "jit.h"
#include "llvm/Support/Casting.h"
#include <set>

std::unique_ptr<llvm::LLVMContext> g_ctx;
std::unique_ptr<llvm::Module> g_mod;
std::unique_ptr<llvm::IRBuilder<>> g_ir;
std::map<std::string, double> g_globals;
std::set<std::string> g_globals_immutable;
bool g_repl_mode = false;
bool g_bounds_err = false;
std::jmp_buf g_bounds_jmp;
std::map<std::string, std::string> g_var_types;
std::map<std::string, int> g_var_lens;
std::map<std::string, llvm::AllocaInst *> g_vars;
std::unique_ptr<llvm::orc::NablaJIT> g_jit;
std::map<std::string, std::unique_ptr<ProtoNode>> g_protos;
std::map<std::string, int> g_op_prec;

llvm::BasicBlock *g_break_bb = nullptr;
llvm::BasicBlock *g_cont_bb = nullptr;

std::unique_ptr<llvm::DIBuilder> g_dbg;
DiState g_di;

std::map<std::string,
         std::vector<std::pair<std::string, std::unique_ptr<TypeNode>>>>
    g_struct_types;

std::map<std::string, std::vector<int>> g_var_dims;

void setup_mod()
{
    g_ctx = std::make_unique<llvm::LLVMContext>();
    g_mod = std::make_unique<llvm::Module>("nabla", *g_ctx);
    g_mod->setDataLayout(g_jit->getDataLayout());
    g_ir = std::make_unique<llvm::IRBuilder<>>(*g_ctx);
}

void g_prec_init()
{
    g_op_prec["="] = 2;
    g_op_prec["||"] = 5;
    g_op_prec["&&"] = 10;
    g_op_prec["=="] = 20;
    g_op_prec["!="] = 20;
    g_op_prec["<"] = 25;
    g_op_prec["<="] = 25;
    g_op_prec[">"] = 25;
    g_op_prec[">="] = 25;
    g_op_prec["+"] = 30;
    g_op_prec["-"] = 30;
    g_op_prec["*"] = 40;
    g_op_prec["/"] = 40;
    g_op_prec["@"] = 40;
    g_op_prec["**"] = 50;
    g_op_prec["'"] = 80;
    g_op_prec["["] = 80;
}
