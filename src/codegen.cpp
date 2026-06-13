#include "codegen.h"
#include "constants.h"
#include "context.h"
#include "jit.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <cstdio>
#include <map>

static std::map<std::string, llvm::StructType *> g_llvm_structs;
static std::map<std::string, std::vector<int>> g_array_op_result_dims;

llvm::DIType *DiState::dbl_ref()
{
    if (dbl_ty)
        return dbl_ty;
    dbl_ty = g_dbg->createBasicType("double", 64, llvm::dwarf::DW_ATE_float);
    return dbl_ty;
}

void DiState::emit_pos(Node *n)
{
    if (!n)
    {
        g_ir->SetCurrentDebugLocation(llvm::DebugLoc());
        return;
    }
    llvm::DIScope *s;
    if (scopes.empty())
        s = cu;
    else
        s = scopes.back();
    g_ir->SetCurrentDebugLocation(
        llvm::DILocation::get(s->getContext(), n->line(), n->col(), s));
}

static llvm::DISubroutineType *mk_func_ty(unsigned n)
{
    llvm::SmallVector<llvm::Metadata *, 8> tys;
    auto *d = g_di.dbl_ref();
    tys.push_back(d);
    for (unsigned i = 0; i < n; ++i)
        tys.push_back(d);
    return g_dbg->createSubroutineType(g_dbg->getOrCreateTypeArray(tys));
}

llvm::Type *llvm_type(const TypeNode *t)
{
    if (!t)
        return llvm::Type::getDoubleTy(*g_ctx);
    switch (t->tag)
    {
    case TypeTag::Int:
        return llvm::Type::getInt64Ty(*g_ctx);
    case TypeTag::Float:
        return llvm::Type::getDoubleTy(*g_ctx);
    case TypeTag::Bool:
        return llvm::Type::getInt1Ty(*g_ctx);
    case TypeTag::Str:
        return llvm::PointerType::get(llvm::Type::getInt8Ty(*g_ctx), 0);
    case TypeTag::Void:
        return llvm::Type::getVoidTy(*g_ctx);
    case TypeTag::Array:
        return llvm::PointerType::get(llvm_type(t->elem.get()), 0);
    case TypeTag::Matrix:
        return llvm::PointerType::get(llvm::Type::getDoubleTy(*g_ctx), 0);
    case TypeTag::Named:
    {
        auto it = g_llvm_structs.find(t->name);
        if (it != g_llvm_structs.end())
            return llvm::PointerType::get(it->second, 0);
        return llvm::Type::getDoubleTy(*g_ctx);
    }
    case TypeTag::Struct:
    {
        std::vector<llvm::Type *> members;
        for (auto &f : t->fields)
            members.push_back(llvm_type(f.second.get()));
        return llvm::StructType::create(*g_ctx, members, "anon");
    }
    }
    return llvm::Type::getDoubleTy(*g_ctx);
}

llvm::Function *find_func(const std::string &name)
{
    if (auto *f = g_mod->getFunction(name))
        return f;
    auto it = g_protos.find(name);
    if (it != g_protos.end())
        return it->second->compile();
    return nullptr;
}

llvm::AllocaInst *entry_slot(llvm::Function *f, llvm::StringRef name,
                             llvm::Type *ty)
{
    llvm::IRBuilder<> tmp(&f->getEntryBlock(), f->getEntryBlock().begin());
    return tmp.CreateAlloca(ty, nullptr, name);
}

static bool is_fp(llvm::Value *v)
{
    return v->getType()->isDoubleTy() || v->getType()->isFloatTy();
}

static bool is_array_type(const std::string &type)
{
    return !type.empty() && type.front() == '[';
}

static bool is_pointer_like_type(const std::string &type)
{
    return type == TYPE_STRING || is_array_type(type) ||
           type.rfind("matrix", 0) == 0;
}

static llvm::Value *promote_to_double(llvm::Value *v)
{
    if (v->getType()->isDoubleTy())
        return v;
    if (v->getType()->isPointerTy())
    {
        auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
        return g_ir->CreateSIToFP(g_ir->CreatePtrToInt(v, i64, "ptri"),
                                  llvm::Type::getDoubleTy(*g_ctx), "i2d");
    }
    if (v->getType()->isIntegerTy(1))
        return g_ir->CreateUIToFP(v, llvm::Type::getDoubleTy(*g_ctx), "b2f");
    if (v->getType()->isIntegerTy())
        return g_ir->CreateSIToFP(v, llvm::Type::getDoubleTy(*g_ctx), "i2d");
    return v;
}

static llvm::Value *to_i64_index(llvm::Value *v)
{
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    if (v->getType()->isIntegerTy(64))
        return v;
    if (v->getType()->isIntegerTy())
        return g_ir->CreateSExtOrTrunc(v, i64, "idx64");
    if (v->getType()->isDoubleTy())
        return g_ir->CreateFPToSI(v, i64, "idx64");
    return v;
}

static int dims_to_len(const std::vector<int> &dims);
static llvm::Value *err_v(const char *msg);
static std::vector<int> get_node_dims(Node *n);

static int get_indexable_len(Node *n)
{
    if (auto *arr = dynamic_cast<ArrNode *>(n))
        return (int)arr->size();
    if (auto *str = dynamic_cast<StrNode *>(n))
        return (int)str->value().size();
    if (auto *sym = dynamic_cast<SymNode *>(n))
    {
        auto lit = g_var_lens.find(sym->get_name());
        if (lit != g_var_lens.end())
            return lit->second;
        auto ty = g_var_types.find(sym->get_name());
        if (ty != g_var_types.end() && ty->second == TYPE_STRING)
            return -1;
        auto dim = g_var_dims.find(sym->get_name());
        if (dim != g_var_dims.end())
            return dims_to_len(dim->second);
    }
    return -1;
}

static llvm::Value *to_bool_cond(llvm::Value *v)
{
    if (v->getType()->isDoubleTy())
        return g_ir->CreateFCmpONE(
            v, llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0)), "f2b");
    if (v->getType()->isIntegerTy(1))
        return v;
    if (v->getType()->isIntegerTy())
        return g_ir->CreateICmpNE(v, llvm::ConstantInt::get(v->getType(), 0),
                                  "i2b");
    return v;
}

static llvm::Value *emit_bounds_error(const char *kind, llvm::Value *idx_i64,
                                      llvm::Value *len_i64)
{
    auto *err_fn = g_mod->getFunction("nabla_bounds_error");
    auto *kind_str = g_ir->CreateGlobalStringPtr(kind, "kind");
    g_ir->CreateCall(err_fn, {kind_str, idx_i64, len_i64});
    return g_ir->CreateUnreachable();
}

static llvm::BasicBlock *emit_div_zero_guard(llvm::Value *den,
                                             const char *label)
{
    auto *fn = g_ir->GetInsertBlock()->getParent();
    auto *err_bb =
        llvm::BasicBlock::Create(*g_ctx, std::string(label) + ".div_zero", fn);
    auto *ok_bb =
        llvm::BasicBlock::Create(*g_ctx, std::string(label) + ".div_ok", fn);

    llvm::Value *is_zero = nullptr;
    if (den->getType()->isIntegerTy())
    {
        is_zero = g_ir->CreateICmpEQ(
            den, llvm::ConstantInt::get(den->getType(), 0), "div_zero");
    }
    else if (den->getType()->isDoubleTy())
    {
        is_zero = g_ir->CreateFCmpOEQ(
            den, llvm::ConstantFP::get(den->getType(), 0.0), "div_zero");
    }
    else
    {
        return nullptr;
    }

    g_ir->CreateCondBr(is_zero, err_bb, ok_bb);

    g_ir->SetInsertPoint(err_bb);
    auto *err_fn = g_mod->getFunction("nabla_div_zero_error");
    g_ir->CreateCall(err_fn, {});
    g_ir->CreateUnreachable();

    g_ir->SetInsertPoint(ok_bb);
    return ok_bb;
}

static llvm::Value *emit_dims_ptr(const std::vector<int> &dims,
                                  const char *name)
{
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *arr_ty = llvm::ArrayType::get(i64, dims.size());
    auto *slot = g_ir->CreateAlloca(arr_ty, nullptr, name);
    auto *zero = llvm::ConstantInt::get(i64, 0);
    for (size_t i = 0; i < dims.size(); ++i)
    {
        auto *gep = g_ir->CreateGEP(
            arr_ty, slot, {zero, llvm::ConstantInt::get(i64, i)}, "dim_gep");
        g_ir->CreateStore(llvm::ConstantInt::get(i64, dims[i]), gep);
    }
    return g_ir->CreateGEP(arr_ty, slot, {zero, zero}, "dims_ptr");
}

static llvm::Value *
emit_row_major_offset(const std::vector<int> &shape,
                      const std::vector<llvm::Value *> &idxs, const char *kind)
{
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    if (shape.empty() || idxs.empty() || shape.size() != idxs.size())
        return err_v("matrix indexing requires one index per dimension");

    llvm::Value *offset = nullptr;
    for (size_t i = 0; i < idxs.size(); ++i)
    {
        auto *idx = to_i64_index(idxs[i]);
        if (!idx)
            return nullptr;
        if (shape[i] > 0)
        {
            auto *dim = llvm::ConstantInt::get(i64, shape[i]);
            auto *neg =
                g_ir->CreateICmpSLT(idx, llvm::ConstantInt::get(i64, 0), "neg");
            auto *too_big = g_ir->CreateICmpSGE(idx, dim, "ob");
            auto *oob = g_ir->CreateOr(neg, too_big, "oob");
            auto *fn = g_ir->GetInsertBlock()->getParent();
            auto *ok_bb = llvm::BasicBlock::Create(*g_ctx, "ok", fn);
            auto *err_bb = llvm::BasicBlock::Create(*g_ctx, "err", fn);
            g_ir->CreateCondBr(oob, err_bb, ok_bb);
            g_ir->SetInsertPoint(err_bb);
            emit_bounds_error(kind, idx, dim);
            g_ir->SetInsertPoint(ok_bb);
        }

        if (!offset)
            offset = idx;
        else
        {
            auto *dim = llvm::ConstantInt::get(i64, shape[i]);
            offset = g_ir->CreateAdd(g_ir->CreateMul(offset, dim, "idx_mul"),
                                     idx, "idx_off");
        }
    }
    return offset;
}

struct LoopBuilder
{
    llvm::Function *fn;
    llvm::BasicBlock *entry;
    llvm::BasicBlock *loop;
    llvm::BasicBlock *exit;
};

static LoopBuilder start_loop(const char *name)
{
    LoopBuilder lb;
    lb.fn = g_ir->GetInsertBlock()->getParent();
    lb.entry = g_ir->GetInsertBlock();
    lb.loop = llvm::BasicBlock::Create(*g_ctx, name, lb.fn);
    lb.exit = llvm::BasicBlock::Create(*g_ctx, name);
    g_ir->CreateBr(lb.loop);
    g_ir->SetInsertPoint(lb.loop);
    return lb;
}

static llvm::PHINode *loop_index(LoopBuilder &lb, const char *name)
{
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *i = g_ir->CreatePHI(i64, 2, name);
    i->addIncoming(llvm::ConstantInt::get(i64, 0), lb.entry);
    return i;
}

static void loop_next(LoopBuilder &lb, llvm::PHINode *i, llvm::Value *count,
                      const char *inc_name, const char *cmp_name)
{
    auto *one = llvm::ConstantInt::get(i->getType(), 1);
    auto *in = g_ir->CreateAdd(i, one, inc_name);
    i->addIncoming(in, lb.loop);
    auto *ic = g_ir->CreateICmpSLT(in, count, cmp_name);
    g_ir->CreateCondBr(ic, lb.loop, lb.exit);
}

static void end_loop(LoopBuilder &lb)
{
    lb.fn->insert(lb.fn->end(), lb.exit);
    g_ir->SetInsertPoint(lb.exit);
}

static llvm::Value *err_v(const char *msg)
{
    fprintf(stderr, "Error: %s\n", msg);
    return nullptr;
}

static llvm::AllocaInst *find_struct_slot(llvm::Value *ptr,
                                          const std::string &field,
                                          llvm::StructType *&st, unsigned &idx)
{
    for (auto &[name, def] : g_struct_types)
    {
        auto *lst = g_llvm_structs[name];
        if (!lst)
            continue;
        if (ptr->getType() != llvm::PointerType::get(lst, 0))
            continue;
        st = lst;
        for (unsigned i = 0; i < def.size(); ++i)
        {
            if (def[i].first == field)
            {
                idx = i;
                return nullptr;
            }
        }
    }
    return nullptr;
}

llvm::Value *NumNode::compile()
{
    g_di.emit_pos(this);
    return llvm::ConstantFP::get(*g_ctx, llvm::APFloat(val));
}

llvm::Value *IntNode::compile()
{
    g_di.emit_pos(this);
    return llvm::ConstantInt::get(llvm::Type::getInt64Ty(*g_ctx), val, true);
}

llvm::Value *StrNode::compile()
{
    g_di.emit_pos(this);
    return g_ir->CreateGlobalStringPtr(val, "str");
}

llvm::Value *BoolNode::compile()
{
    g_di.emit_pos(this);
    return llvm::ConstantInt::get(llvm::Type::getInt1Ty(*g_ctx), val);
}

llvm::Value *SymNode::compile()
{
    auto it = g_vars.find(name);
    if (it != g_vars.end())
    {
        g_di.emit_pos(this);
        return g_ir->CreateLoad(it->second->getAllocatedType(), it->second,
                                name.c_str());
    }
    if (g_globals.count(name))
    {
        g_di.emit_pos(this);
        auto *get_global = g_mod->getFunction("nabla_get_global");
        auto *name_str = g_ir->CreateGlobalStringPtr(name, "gname");
        auto *val = g_ir->CreateCall(get_global, name_str, name.c_str());
        auto vt = g_var_types.find(name);
        if (vt != g_var_types.end() && is_pointer_like_type(vt->second))
        {
            auto *i64_ty = llvm::Type::getInt64Ty(*g_ctx);
            auto *i64_v = g_ir->CreateFPToSI(val, i64_ty, "d2i");
            return g_ir->CreateIntToPtr(
                i64_v, llvm::PointerType::get(*g_ctx, 0), "i2p");
        }
        return val;
    }
    return err_v("unknown variable");
}

llvm::Value *UnaryNode::compile()
{
    auto *ov = operand->compile();
    if (!ov)
        return nullptr;
    if (op == "-")
    {
        if (is_fp(ov))
            return g_ir->CreateFNeg(ov, "neg");
        return g_ir->CreateNeg(ov, "neg");
    }
    if (op == "!")
    {
        if (is_fp(ov))
        {
            auto *z = llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0));
            return g_ir->CreateFCmpOEQ(ov, z, "not");
        }
        auto *z = llvm::ConstantInt::get(ov->getType(), 0);
        return g_ir->CreateICmpEQ(ov, z, "not");
    }
    auto *f = find_func(op);
    if (!f)
        return err_v("unknown unary operator");
    g_di.emit_pos(this);
    return g_ir->CreateCall(f, ov, "unop");
}

static bool both_int(llvm::Value *l, llvm::Value *r)
{
    return l->getType()->isIntegerTy(64) && r->getType()->isIntegerTy(64);
}

static void promote_pair(llvm::Value *&l, llvm::Value *&r)
{
    if (l->getType()->isIntegerTy(64) && r->getType()->isDoubleTy())
        l = g_ir->CreateSIToFP(l, llvm::Type::getDoubleTy(*g_ctx), "i2f");
    else if (l->getType()->isDoubleTy() && r->getType()->isIntegerTy(64))
        r = g_ir->CreateSIToFP(r, llvm::Type::getDoubleTy(*g_ctx), "i2f");
}

static llvm::Value *int_to_bool_i1(llvm::Value *v)
{
    if (v->getType()->isIntegerTy(1))
        return v;
    if (v->getType()->isIntegerTy())
        return g_ir->CreateICmpNE(v, llvm::ConstantInt::get(v->getType(), 0),
                                  "i2b");
    if (v->getType()->isDoubleTy())
        return g_ir->CreateFCmpONE(
            v, llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0)), "f2b");
    return v;
}

llvm::Value *BinaryNode::compile()
{
    g_di.emit_pos(this);

    if (op == "=")
    {
        if (auto *sym = dynamic_cast<SymNode *>(lhs.get()))
        {
            auto *val = rhs->compile();
            if (!val)
                return nullptr;
            auto it = g_vars.find(sym->get_name());
            if (it != g_vars.end())
            {
                auto *dst_ty = it->second->getAllocatedType();
                auto var_type = g_var_types.find(sym->get_name());
                if (var_type != g_var_types.end() &&
                    is_pointer_like_type(var_type->second) &&
                    !val->getType()->isPointerTy())
                {
                    std::string msg =
                        "cannot assign non-matrix value to matrix variable '" +
                        sym->get_name() + "'";
                    return err_v(msg.c_str());
                }
                if (val->getType() != dst_ty)
                {
                    if (val->getType()->isDoubleTy() && dst_ty->isIntegerTy(64))
                        val = g_ir->CreateFPToSI(val, dst_ty, "f2i");
                    else if (val->getType()->isIntegerTy(64) &&
                             dst_ty->isDoubleTy())
                        val = g_ir->CreateSIToFP(val, dst_ty, "i2f");
                    else if (val->getType()->isDoubleTy() &&
                             dst_ty->isIntegerTy(1))
                        val = g_ir->CreateFCmpONE(
                            val,
                            llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0)),
                            "f2b");
                    else if (val->getType()->isIntegerTy(1) &&
                             dst_ty->isDoubleTy())
                        val = g_ir->CreateUIToFP(val, dst_ty, "b2f");
                }
                g_ir->CreateStore(val, it->second);
                return val;
            }
            if (g_globals.count(sym->get_name()))
            {
                if (g_globals_immutable.count(sym->get_name()))
                    return err_v("cannot assign to immutable variable");
                auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
                auto *set_global = g_mod->getFunction("nabla_set_global");

                {
                    auto it = g_var_types.find(sym->get_name());
                    if (it != g_var_types.end())
                    {
                        auto &var_type = it->second;
                        if (is_pointer_like_type(var_type) &&
                            !val->getType()->isPointerTy())
                        {
                            std::string msg = "cannot assign non-matrix value "
                                              "to matrix variable '" +
                                              sym->get_name() + "'";
                            return err_v(msg.c_str());
                        }
                        if (var_type == TYPE_INT &&
                            !val->getType()->isIntegerTy(64))
                        {
                            std::string msg =
                                "cannot assign non-int value to int "
                                "variable '" +
                                sym->get_name() + "'";
                            return err_v(msg.c_str());
                        }
                        if (var_type == TYPE_BOOL &&
                            !val->getType()->isIntegerTy(1))
                        {
                            std::string msg =
                                "cannot assign non-bool value to bool "
                                "variable '" +
                                sym->get_name() + "'";
                            return err_v(msg.c_str());
                        }
                        if (var_type == TYPE_STRING &&
                            !val->getType()->isPointerTy())
                        {
                            std::string msg =
                                "cannot assign non-string value to "
                                "string variable '" +
                                sym->get_name() + "'";
                            return err_v(msg.c_str());
                        }
                        if (var_type == TYPE_FLOAT)
                        {
                            if (val->getType()->isIntegerTy(64))
                                val = g_ir->CreateSIToFP(val, dbl, "i2d");
                            else if (!val->getType()->isDoubleTy())
                            {
                                std::string msg =
                                    "cannot assign non-float value to float "
                                    "variable '" +
                                    sym->get_name() + "'";
                                return err_v(msg.c_str());
                            }
                        }
                    }
                }

                if (val->getType() != dbl)
                    val = promote_to_double(val);
                auto *name_str =
                    g_ir->CreateGlobalStringPtr(sym->get_name(), "gname");
                auto *stored = g_ir->CreateCall(set_global, {name_str, val});
                auto *decl_fn = g_mod->getFunction("nabla_print_decl");
                if (decl_fn)
                {
                    auto it = g_var_types.find(sym->get_name());
                    auto *type_str = g_ir->CreateGlobalStringPtr(
                        it != g_var_types.end() ? it->second.c_str()
                                                : TYPE_FLOAT,
                        "tname");
                    g_ir->CreateCall(decl_fn, {name_str, type_str, stored});
                }
                return stored;
            }
            return err_v("unknown variable");
        }
        if (auto *idx = dynamic_cast<IdxNode *>(lhs.get()))
        {
            auto *target_val = idx->get_target()->compile();
            auto *store_val = rhs->compile();
            if (!target_val || !store_val)
                return nullptr;
            auto *target = idx->get_target();
            if (dynamic_cast<StrNode *>(target))
                return err_v("cannot assign to string index");
            if (store_val->getType()->isIntegerTy())
                store_val = promote_to_double(store_val);

            bool is_matrix = false;
            if (auto *sym = dynamic_cast<SymNode *>(target))
                is_matrix = g_var_dims.count(sym->get_name()) > 0;
            auto shape = get_node_dims(target);
            std::vector<llvm::Value *> idx_vals;
            idx_vals.reserve(idx->get_indices().size());
            for (auto &ind : idx->get_indices())
            {
                auto *iv = ind->compile();
                if (!iv)
                    return nullptr;
                idx_vals.push_back(iv);
            }
            auto *offset = emit_row_major_offset(
                shape, idx_vals, is_matrix ? "matrix" : "array");
            if (!offset)
                return nullptr;

            auto *gep = g_ir->CreateGEP(llvm::Type::getDoubleTy(*g_ctx),
                                        target_val, offset, "idxm");
            g_ir->CreateStore(store_val, gep);
            return store_val;
        }
        return err_v("assignment target must be a variable");
    }

    auto *l = lhs->compile();
    auto *r = rhs->compile();
    if (!l || !r)
        return nullptr;

    auto *dbl_ty = llvm::Type::getDoubleTy(*g_ctx);

    auto is_array_node = [](Node *n) -> bool
    {
        if (dynamic_cast<ArrNode *>(n))
            return true;
        if (auto *sym = dynamic_cast<SymNode *>(n))
        {
            auto it = g_var_types.find(sym->get_name());
            if (it != g_var_types.end() && is_array_type(it->second))
                return true;
        }
        return false;
    };

    auto compile_array_op = [&](llvm::Value *l_ptr, llvm::Value *r_ptr,
                                const char *op_name) -> llvm::Value *
    {
        int sz = -1;
        if (auto *arr = dynamic_cast<ArrNode *>(lhs.get()))
            sz = (int)arr->size();
        else if (auto *sym = dynamic_cast<SymNode *>(lhs.get()))
        {
            auto it = g_var_lens.find(sym->get_name());
            if (it != g_var_lens.end())
                sz = it->second;
            else
            {
                auto dit = g_var_dims.find(sym->get_name());
                if (dit != g_var_dims.end())
                    sz = dims_to_len(dit->second);
            }
        }

        if (sz <= 0)
            return err_v("cannot determine array size for operation");

        auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
        auto *sz_c = llvm::ConstantInt::get(i64, sz);
        auto *res_arr = g_ir->CreateAlloca(dbl_ty, sz_c, "arr_op_res");

        auto lb = start_loop("arr_op_loop");
        auto *i = loop_index(lb, "arr_op_i");
        auto *l_elem = g_ir->CreateLoad(
            dbl_ty, g_ir->CreateGEP(dbl_ty, l_ptr, i, "l_gep"), "l_elem");
        auto *r_elem = g_ir->CreateLoad(
            dbl_ty, g_ir->CreateGEP(dbl_ty, r_ptr, i, "r_gep"), "r_elem");

        llvm::Value *res_elem = nullptr;
        if (op_name == std::string("+"))
            res_elem = g_ir->CreateFAdd(l_elem, r_elem, "elem_add");
        else if (op_name == std::string("-"))
            res_elem = g_ir->CreateFSub(l_elem, r_elem, "elem_sub");
        else if (op_name == std::string("*"))
            res_elem = g_ir->CreateFMul(l_elem, r_elem, "elem_mul");
        else if (op_name == std::string("/"))
        {
            if (!emit_div_zero_guard(r_elem, "arr_div"))
                return err_v("unsupported array operation");
            res_elem = g_ir->CreateFDiv(l_elem, r_elem, "elem_div");
        }
        else
            return err_v("unsupported array operation");

        auto *res_gep = g_ir->CreateGEP(dbl_ty, res_arr, i, "res_gep");
        g_ir->CreateStore(res_elem, res_gep);

        loop_next(lb, i, sz_c, "arr_op_in", "arr_op_ic");
        end_loop(lb);

        auto *dup_arr_fn = g_mod->getFunction("nabla_dup_array");
        if (!dup_arr_fn)
            return err_v("missing array duplication runtime");

        auto *dbl_ptr_ty = llvm::PointerType::get(dbl_ty, 0);
        auto *casted_res =
            g_ir->CreateBitCast(res_arr, dbl_ptr_ty, "arr_dup_src");
        return g_ir->CreateCall(dup_arr_fn, {casted_res, sz_c}, "arr_ret");
    };

    if (op == "+" && l->getType()->isPointerTy() && r->getType()->isPointerTy())
    {
        if (is_array_node(lhs.get()) && is_array_node(rhs.get()))
            return compile_array_op(l, r, "+");
        auto *strcat = g_mod->getFunction("nabla_strcat");
        return g_ir->CreateCall(strcat, {l, r}, "cat");
    }

    if (op == "-" && l->getType()->isPointerTy() &&
        r->getType()->isPointerTy() && is_array_node(lhs.get()) &&
        is_array_node(rhs.get()))
        return compile_array_op(l, r, "-");

    if (op == "*" && l->getType()->isPointerTy() &&
        r->getType()->isPointerTy() && is_array_node(lhs.get()) &&
        is_array_node(rhs.get()))
        return compile_array_op(l, r, "*");

    if (op == "/" && l->getType()->isPointerTy() &&
        r->getType()->isPointerTy() && is_array_node(lhs.get()) &&
        is_array_node(rhs.get()))
        return compile_array_op(l, r, "/");

    if (op == "||")
    {
        auto *lb = int_to_bool_i1(l);
        auto *rb = int_to_bool_i1(r);
        return g_ir->CreateOr(lb, rb, "orv");
    }
    if (op == "&&")
    {
        auto *lb = int_to_bool_i1(l);
        auto *rb = int_to_bool_i1(r);
        return g_ir->CreateAnd(lb, rb, "andv");
    }

    if (both_int(l, r))
    {
        if (op == "+")
            return g_ir->CreateAdd(l, r, "add");
        if (op == "-")
            return g_ir->CreateSub(l, r, "sub");
        if (op == "*")
            return g_ir->CreateMul(l, r, "mul");
        if (op == "/")
        {
            auto *ok_bb = emit_div_zero_guard(r, "int_div");
            if (!ok_bb)
                return err_v("unsupported division operand");
            auto *res = g_ir->CreateSDiv(l, r, "div");
            auto *cont_bb = llvm::BasicBlock::Create(
                *g_ctx, "int_div.cont", g_ir->GetInsertBlock()->getParent());
            g_ir->CreateBr(cont_bb);
            g_ir->SetInsertPoint(cont_bb);
            auto *phi = g_ir->CreatePHI(res->getType(), 1, "div_phi");
            phi->addIncoming(res, ok_bb);
            return phi;
        }
    }
    else
    {
        promote_pair(l, r);
        if (op == "+")
            return g_ir->CreateFAdd(l, r, "add");
        if (op == "-")
            return g_ir->CreateFSub(l, r, "sub");
        if (op == "*")
            return g_ir->CreateFMul(l, r, "mul");
        if (op == "/")
        {
            auto *ok_bb = emit_div_zero_guard(r, "float_div");
            if (!ok_bb)
                return err_v("unsupported division operand");
            auto *res = g_ir->CreateFDiv(l, r, "div");
            auto *cont_bb = llvm::BasicBlock::Create(
                *g_ctx, "float_div.cont", g_ir->GetInsertBlock()->getParent());
            g_ir->CreateBr(cont_bb);
            g_ir->SetInsertPoint(cont_bb);
            auto *phi = g_ir->CreatePHI(res->getType(), 1, "div_phi");
            phi->addIncoming(res, ok_bb);
            return phi;
        }
    }

    llvm::Value *cmp = nullptr;
    if (both_int(l, r))
    {
        if (op == "<")
            cmp = g_ir->CreateICmpSLT(l, r, "cmp");
        else if (op == "<=")
            cmp = g_ir->CreateICmpSLE(l, r, "cmp");
        else if (op == ">")
            cmp = g_ir->CreateICmpSGT(l, r, "cmp");
        else if (op == ">=")
            cmp = g_ir->CreateICmpSGE(l, r, "cmp");
        else if (op == "==")
            cmp = g_ir->CreateICmpEQ(l, r, "cmp");
        else if (op == "!=")
            cmp = g_ir->CreateICmpNE(l, r, "cmp");
    }
    else
    {
        promote_pair(l, r);
        if (op == "<")
            cmp = g_ir->CreateFCmpOLT(l, r, "cmp");
        else if (op == "<=")
            cmp = g_ir->CreateFCmpOLE(l, r, "cmp");
        else if (op == ">")
            cmp = g_ir->CreateFCmpOGT(l, r, "cmp");
        else if (op == ">=")
            cmp = g_ir->CreateFCmpOGE(l, r, "cmp");
        else if (op == "==")
            cmp = g_ir->CreateFCmpOEQ(l, r, "cmp");
        else if (op == "!=")
            cmp = g_ir->CreateFCmpONE(l, r, "cmp");
    }

    if (cmp)
        return cmp;

    if (op == "**")
    {
        l = promote_to_double(l);
        r = promote_to_double(r);
        auto *pow_intr = llvm::Intrinsic::getOrInsertDeclaration(
            g_mod.get(), llvm::Intrinsic::pow, {dbl_ty});
        return g_ir->CreateCall(pow_intr, {l, r}, "pow");
    }

    auto *f = find_func("binary" + op);
    if (!f)
        return err_v("unknown operator");
    llvm::Value *args[] = {l, r};
    return g_ir->CreateCall(f, args, "binop");
}

static int get_const_int(Node *n)
{
    if (auto *in = dynamic_cast<IntNode *>(n))
        return (int)in->value();
    if (auto *nn = dynamic_cast<NumNode *>(n))
        return (int)nn->value();
    return -1;
}

static std::vector<int> get_node_dims(Node *n)
{
    if (!n)
        return {};
    if (auto *sym = dynamic_cast<SymNode *>(n))
    {
        auto it = g_var_dims.find(sym->get_name());
        if (it != g_var_dims.end())
            return it->second;
    }
    if (auto *arr = dynamic_cast<ArrNode *>(n))
        return {(int)arr->size()};
    if (auto *mm = dynamic_cast<MatMulNode *>(n))
        return {mm->get_rows(), mm->get_cols()};
    if (auto *tp = dynamic_cast<TransposeNode *>(n))
        return {tp->get_rows(), tp->get_cols()};
    if (auto *gr = dynamic_cast<GradNode *>(n))
        return {gr->get_rows(), gr->get_cols()};
    if (auto *jac = dynamic_cast<JacobianNode *>(n))
        return {jac->get_rows(), jac->get_cols()};
    if (auto *bin = dynamic_cast<BinaryNode *>(n))
    {
        auto op = bin->get_op();
        if (op == "+" || op == "-" || op == "*" || op == "/")
            return get_node_dims(bin->get_lhs());
    }
    if (auto *call = dynamic_cast<CallNode *>(n))
    {
        const auto &cn = call->get_callee();
        const auto &ca = call->get_args();
        if (cn == "eye")
        {
            int d = ca.empty() ? -1 : get_const_int(ca[0].get());
            if (d > 0)
                return {d, d};
        }
        else if (cn == "zeros" || cn == "ones")
        {
            if (ca.size() >= 2)
            {
                int r = get_const_int(ca[0].get());
                int c = get_const_int(ca[1].get());
                if (r > 0 && c > 0)
                    return {r, c};
            }
            else
            {
                int d = ca.empty() ? -1 : get_const_int(ca[0].get());
                if (d > 0)
                    return {1, d};
            }
        }
        else if (cn == "linspace")
        {
            int d = ca.size() < 3 ? -1 : get_const_int(ca[2].get());
            if (d > 0)
                return {1, d};
        }
        else if (cn == "solve")
        {
            if (ca.size() >= 2)
            {
                auto dims = get_node_dims(ca[1].get());
                if (dims.size() == 1 && dims[0] > 0)
                    return {dims[0]};
                if (dims.size() >= 2 && dims[0] > 0 && dims[1] == 1)
                    return {dims[0], 1};
            }
        }
        else if (cn == "inv")
        {
            if (!ca.empty())
            {
                auto dims = get_node_dims(ca[0].get());
                if (dims.size() >= 2 && dims[0] > 0 && dims[0] == dims[1])
                    return dims;
            }
        }
        else if (cn == "eig")
        {
            if (!ca.empty())
            {
                auto dims = get_node_dims(ca[0].get());
                if (!dims.empty() && dims[0] > 0)
                    return {1, dims[0]};
            }
        }
    }
    return {};
}

static void infer_dims(Node *init, const std::string &name)
{
    auto dims = get_node_dims(init);
    if (!dims.empty() && dims_to_len(dims) > 0)
        g_var_dims[name] = dims;
}

static llvm::Value *compile_sqrt(const std::vector<std::unique_ptr<Node>> &args)
{
    auto *v = args[0]->compile();
    if (!v)
        return nullptr;
    auto *sqrt_intr = llvm::Intrinsic::getOrInsertDeclaration(
        g_mod.get(), llvm::Intrinsic::sqrt, {llvm::Type::getDoubleTy(*g_ctx)});
    return g_ir->CreateCall(sqrt_intr, v, "sqrt");
}

static llvm::Value *compile_trig(const std::vector<std::unique_ptr<Node>> &args,
                                 llvm::Intrinsic::ID intrinsic_id,
                                 const char *name)
{
    auto *v = args[0]->compile();
    if (!v)
        return nullptr;
    v = promote_to_double(v);
    auto *intr = llvm::Intrinsic::getOrInsertDeclaration(
        g_mod.get(), intrinsic_id, {llvm::Type::getDoubleTy(*g_ctx)});
    return g_ir->CreateCall(intr, v, name);
}

static llvm::Value *compile_input()
{
    auto *input_fn = g_mod->getFunction("nabla_input");
    return g_ir->CreateCall(input_fn, {}, "input");
}

static llvm::Value *
compile_typeof(const std::vector<std::unique_ptr<Node>> &args)
{
    auto *ps = g_mod->getFunction("nabla_print_str");
    auto *putch = g_mod->getFunction("putchard");
    std::string type_name = TYPE_FLOAT;

    if (!args.empty())
    {
        if (auto *sym = dynamic_cast<SymNode *>(args[0].get()))
        {
            auto it = g_var_types.find(sym->get_name());
            if (it != g_var_types.end())
                type_name = it->second;
        }
        if (dynamic_cast<ArrNode *>(args[0].get()))
            type_name = "[float]";
        auto *v = args[0]->compile();
        if (v)
        {
            if (is_pointer_like_type(type_name))
            {
                // keep array/matrix/string type
            }
            else if (v->getType()->isPointerTy())
                type_name = TYPE_STRING;
            else if (v->getType()->isIntegerTy(1))
                type_name = TYPE_BOOL;
            else if (v->getType()->isIntegerTy(64))
                type_name = TYPE_INT;
        }
        auto *type_str = g_ir->CreateGlobalStringPtr(type_name, "tname");
        g_ir->CreateCall(ps, type_str);
        g_ir->CreateCall(putch,
                         llvm::ConstantFP::get(*g_ctx, llvm::APFloat(10.0)));
    }
    return llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0));
}

static llvm::Value *
compile_print(const std::vector<std::unique_ptr<Node>> &args)
{
    auto *putch = g_mod->getFunction("putchard");
    auto *pd = g_mod->getFunction("printd");
    auto *ps = g_mod->getFunction("nabla_print_str");

    if (!args.empty())
    {
        auto *v = args[0]->compile();
        if (!v)
            return nullptr;
        if (v->getType()->isPointerTy())
        {
            auto dims = get_node_dims(args[0].get());
            if (!dims.empty() && dims_to_len(dims) > 0)
            {
                auto *print_nd = g_mod->getFunction("nabla_print_ndmatrix");
                auto *dims_ptr = emit_dims_ptr(dims, "prshape");
                auto *ndims = llvm::ConstantInt::get(
                    llvm::Type::getInt64Ty(*g_ctx), (uint64_t)dims.size());
                g_ir->CreateCall(print_nd, {v, dims_ptr, ndims});
                return llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0));
            }
            if (auto *sym = dynamic_cast<SymNode *>(args[0].get()))
            {
                auto vt = g_var_types.find(sym->get_name());
                if (vt != g_var_types.end() && is_array_type(vt->second))
                {
                    auto shape = g_var_dims.find(sym->get_name());
                    std::vector<int> dimsv;
                    if (shape != g_var_dims.end())
                        dimsv = shape->second;
                    else
                    {
                        auto len = g_var_lens.find(sym->get_name());
                        if (len != g_var_lens.end() && len->second > 0)
                            dimsv = {len->second};
                    }
                    if (!dimsv.empty() && dims_to_len(dimsv) > 0)
                    {
                        auto *print_nd =
                            g_mod->getFunction("nabla_print_ndmatrix");
                        auto *dims_ptr = emit_dims_ptr(dimsv, "arrshape");
                        auto *ndims = llvm::ConstantInt::get(
                            llvm::Type::getInt64Ty(*g_ctx),
                            (uint64_t)dimsv.size());
                        g_ir->CreateCall(print_nd, {v, dims_ptr, ndims});
                        return llvm::ConstantFP::get(*g_ctx,
                                                     llvm::APFloat(0.0));
                    }
                }
            }
            g_ir->CreateCall(ps, v);
            g_ir->CreateCall(
                putch, llvm::ConstantFP::get(*g_ctx, llvm::APFloat(10.0)));
            return llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0));
        }
        if (v->getType()->isIntegerTy(1))
        {
            auto *true_str = g_ir->CreateGlobalStringPtr("true", "btrue");
            auto *false_str = g_ir->CreateGlobalStringPtr("false", "bfalse");
            auto *sel = g_ir->CreateSelect(v, true_str, false_str, "bstr");
            g_ir->CreateCall(ps, sel);
            g_ir->CreateCall(
                putch, llvm::ConstantFP::get(*g_ctx, llvm::APFloat(10.0)));
            return llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0));
        }
        if (v->getType()->isDoubleTy())
        {
            if (auto *sym = dynamic_cast<SymNode *>(args[0].get()))
            {
                auto it = g_var_types.find(sym->get_name());
                if (it != g_var_types.end() && it->second == TYPE_BOOL)
                {
                    auto *zero =
                        llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0));
                    auto *cmp = g_ir->CreateFCmpONE(v, zero, "bcheck");
                    auto *true_str =
                        g_ir->CreateGlobalStringPtr("true", "btrue");
                    auto *false_str =
                        g_ir->CreateGlobalStringPtr("false", "bfalse");
                    auto *sel =
                        g_ir->CreateSelect(cmp, true_str, false_str, "bstr");
                    g_ir->CreateCall(ps, sel);
                    g_ir->CreateCall(putch, llvm::ConstantFP::get(
                                                *g_ctx, llvm::APFloat(10.0)));
                    return llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0));
                }
                if (it != g_var_types.end() && it->second == TYPE_STRING)
                {
                    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
                    auto *i8ptr = llvm::PointerType::get(*g_ctx, 0);
                    auto *as_int = g_ir->CreateFPToSI(v, i64, "d2i");
                    auto *as_ptr = g_ir->CreateIntToPtr(as_int, i8ptr, "i2p");
                    g_ir->CreateCall(ps, as_ptr);
                    g_ir->CreateCall(putch, llvm::ConstantFP::get(
                                                *g_ctx, llvm::APFloat(10.0)));
                    return llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0));
                }
            }
            g_ir->CreateCall(pd, v);
            g_ir->CreateCall(
                putch, llvm::ConstantFP::get(*g_ctx, llvm::APFloat(10.0)));
            return v;
        }
        if (auto *idx = dynamic_cast<IdxNode *>(args[0].get()))
        {
            auto *target = idx->get_target();
            bool is_str = false;
            bool is_array = false;
            if (auto *sym = dynamic_cast<SymNode *>(target))
            {
                auto it = g_var_types.find(sym->get_name());
                is_str = (it != g_var_types.end() && it->second == TYPE_STRING);
                is_array =
                    (it != g_var_types.end() && is_array_type(it->second));
            }
            else if (dynamic_cast<StrNode *>(target))
                is_str = true;
            else if (dynamic_cast<ArrNode *>(target))
                is_array = true;
            if (is_str)
            {
                auto *chd = g_ir->CreateSIToFP(
                    v, llvm::Type::getDoubleTy(*g_ctx), "chd");
                g_ir->CreateCall(putch, chd);
                g_ir->CreateCall(
                    putch, llvm::ConstantFP::get(*g_ctx, llvm::APFloat(10.0)));
                return llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0));
            }
            if (is_array)
            {
                auto shape = get_node_dims(target);
                if (shape.empty())
                {
                    int len = get_indexable_len(target);
                    if (len > 0)
                        shape = {len};
                }
                if (!shape.empty() && dims_to_len(shape) > 0)
                {
                    auto *print_nd = g_mod->getFunction("nabla_print_ndmatrix");
                    auto *dims_ptr = emit_dims_ptr(shape, "arrshape");
                    auto *ndims = llvm::ConstantInt::get(
                        llvm::Type::getInt64Ty(*g_ctx), (uint64_t)shape.size());
                    g_ir->CreateCall(print_nd, {v, dims_ptr, ndims});
                }
                g_ir->CreateCall(
                    putch, llvm::ConstantFP::get(*g_ctx, llvm::APFloat(10.0)));
                return llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0));
            }
        }
        if (v->getType()->isIntegerTy())
            v = promote_to_double(v);
        g_ir->CreateCall(pd, v);
        g_ir->CreateCall(putch,
                         llvm::ConstantFP::get(*g_ctx, llvm::APFloat(10.0)));
        return v;
    }
    g_ir->CreateCall(putch, llvm::ConstantFP::get(*g_ctx, llvm::APFloat(10.0)));
    return llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0));
}

static int get_size(Node *n, const char *name)
{
    if (auto *sym = dynamic_cast<SymNode *>(n))
    {
        auto it = g_var_dims.find(sym->get_name());
        if (it != g_var_dims.end())
            return dims_to_len(it->second);
    }
    else if (auto *arr = dynamic_cast<ArrNode *>(n))
    {
        return (int)arr->size();
    }
    return -1;
}

static llvm::Value *
compile_eye_zeros_ones(const std::string &callee,
                       const std::vector<std::unique_ptr<Node>> &args)
{
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    auto *n = args[0]->compile();
    if (!n)
        return nullptr;
    auto *ni = g_ir->CreateFPToSI(n, i64, "b_n");
    auto *ni_sq = callee == "eye" ? g_ir->CreateMul(ni, ni, "b_nsq") : ni;
    auto *arr = g_ir->CreateAlloca(dbl, ni_sq, (callee + "_arr").c_str());
    auto lb = start_loop("b_loop");
    auto *i = loop_index(lb, "b_i");
    auto *gep = g_ir->CreateGEP(dbl, arr, i, "b_gep");
    if (callee == "ones")
        g_ir->CreateStore(llvm::ConstantFP::get(dbl, 1.0), gep);
    else
        g_ir->CreateStore(llvm::ConstantFP::get(dbl, 0.0), gep);
    if (callee == "eye")
    {
        auto *diag_cond = g_ir->CreateICmpEQ(
            g_ir->CreateURem(
                i, g_ir->CreateAdd(ni, llvm::ConstantInt::get(i64, 1), "b_np1"),
                "b_rem"),
            llvm::ConstantInt::get(i64, 0), "b_is_diag");
        auto *dbb = llvm::BasicBlock::Create(*g_ctx, "b_diag", lb.fn);
        auto *mbb = llvm::BasicBlock::Create(*g_ctx, "b_merge", lb.fn);
        g_ir->CreateCondBr(diag_cond, dbb, mbb);
        g_ir->SetInsertPoint(dbb);
        g_ir->CreateStore(llvm::ConstantFP::get(dbl, 1.0), gep);
        g_ir->CreateBr(mbb);
        g_ir->SetInsertPoint(mbb);
    }
    auto *one = llvm::ConstantInt::get(i64, 1);
    auto *in = g_ir->CreateAdd(i, one, "b_in");
    i->addIncoming(in, g_ir->GetInsertBlock());
    auto *ic = g_ir->CreateICmpSLT(in, ni_sq, "b_ic");
    g_ir->CreateCondBr(ic, lb.loop, lb.exit);
    end_loop(lb);
    return arr;
}

static llvm::Value *
compile_linspace(const std::vector<std::unique_ptr<Node>> &args)
{
    if (args.size() != 3)
        return err_v("linspace requires exactly 3 arguments: linspace(start, "
                     "end, count)");
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    auto *a = args[0]->compile();
    auto *b = args[1]->compile();
    auto *n = args[2]->compile();
    if (!a || !b || !n)
        return nullptr;
    auto *ni = g_ir->CreateFPToSI(n, i64, "ls_n");
    auto *nd = promote_to_double(n);
    auto *arr = g_ir->CreateAlloca(dbl, ni, "ls_arr");
    auto lb = start_loop("ls_loop");
    auto *i = loop_index(lb, "ls_i");
    auto *idbl = g_ir->CreateSIToFP(i, dbl, "ls_id");
    auto *denom =
        g_ir->CreateFSub(nd, llvm::ConstantFP::get(dbl, 1.0), "ls_nm1");
    auto *div =
        g_ir->CreateFSub(llvm::ConstantFP::get(dbl, 1.0),
                         g_ir->CreateFDiv(idbl, denom, "ls_frac"), "ls_omf");
    auto *val = g_ir->CreateFAdd(
        g_ir->CreateFMul(div, a, "ls_ta"),
        g_ir->CreateFMul(g_ir->CreateFDiv(idbl, denom, "ls_frac2"), b, "ls_tb"),
        "ls_v");
    g_ir->CreateStore(val, g_ir->CreateGEP(dbl, arr, i, "ls_gep"));
    loop_next(lb, i, ni, "ls_in", "ls_ic");
    end_loop(lb);
    return arr;
}

static llvm::Value *
compile_range(const std::vector<std::unique_ptr<Node>> &args)
{
    if (args.size() != 2)
        return err_v("range requires exactly 2 arguments: range(start, end)");
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *a = args[0]->compile();
    auto *b = args[1]->compile();
    if (!a || !b)
        return nullptr;
    auto *ad = promote_to_double(a);
    auto *bd = promote_to_double(b);
    auto *count = g_ir->CreateFPToSI(
        g_ir->CreateFAdd(g_ir->CreateFSub(bd, ad, "r_diff"),
                         llvm::ConstantFP::get(dbl, 1.0), "r_n"),
        i64, "r_ni");
    auto *arr = g_ir->CreateAlloca(dbl, count, "r_arr");
    auto lb = start_loop("r_loop");
    auto *i = loop_index(lb, "r_i");
    g_ir->CreateStore(
        g_ir->CreateFAdd(ad, g_ir->CreateSIToFP(i, dbl, "r_id"), "r_v"),
        g_ir->CreateGEP(dbl, arr, i, "r_gep"));
    loop_next(lb, i, count, "r_in", "r_ic");
    end_loop(lb);

    auto *dup_arr_fn = g_mod->getFunction("nabla_dup_array");
    if (!dup_arr_fn)
        return err_v("missing array duplication runtime");

    auto *dbl_ptr_ty = llvm::PointerType::get(dbl, 0);
    auto *casted_arr = g_ir->CreateBitCast(arr, dbl_ptr_ty, "r_dup_src");
    return g_ir->CreateCall(dup_arr_fn, {casted_arr, count}, "r_ret");
}

static llvm::Value *compile_mean(const std::vector<std::unique_ptr<Node>> &args)
{
    if (args.size() != 1)
        return err_v("mean requires exactly 1 argument: mean(array)");
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    int sz = get_size(args[0].get(), "mean");
    if (sz < 0)
        return err_v("mean: cannot determine size; use type annotations");
    auto *xv = args[0]->compile();
    if (!xv)
        return nullptr;
    auto lb = start_loop("m_loop");
    auto *i = loop_index(lb, "m_i");
    auto *s = g_ir->CreatePHI(dbl, 2, "m_s");
    s->addIncoming(llvm::ConstantFP::get(dbl, 0.0), lb.entry);
    auto *v = g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, xv, i, "m_g"), "m_v");
    auto *sn = g_ir->CreateFAdd(s, v, "m_sn");
    s->addIncoming(sn, lb.loop);
    auto *sz_c = llvm::ConstantInt::get(i64, sz);
    loop_next(lb, i, sz_c, "m_in", "m_ic");
    end_loop(lb);
    return g_ir->CreateFDiv(sn, llvm::ConstantFP::get(dbl, (double)sz), "mean");
}

static llvm::Value *compile_std(const std::vector<std::unique_ptr<Node>> &args)
{
    if (args.size() != 1)
        return err_v("std requires exactly 1 argument: std(array)");
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    int sz = get_size(args[0].get(), "std");
    if (sz < 0)
        return err_v("std: cannot determine size; use type annotations");
    auto *xv = args[0]->compile();
    if (!xv)
        return nullptr;
    auto *fn = g_ir->GetInsertBlock()->getParent();
    auto *zero_i = llvm::ConstantInt::get(i64, 0);
    auto *sz_c = llvm::ConstantInt::get(i64, sz);

    auto lb1 = start_loop("std_s1");
    auto *i1 = loop_index(lb1, "sd_i1");
    auto *s1 = g_ir->CreatePHI(dbl, 2, "sd_s1");
    s1->addIncoming(llvm::ConstantFP::get(dbl, 0.0), lb1.entry);
    auto *v1 =
        g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, xv, i1, "sd_g1"), "sd_v1");
    auto *sn1 = g_ir->CreateFAdd(s1, v1, "sd_sn1");
    s1->addIncoming(sn1, lb1.loop);
    loop_next(lb1, i1, sz_c, "sd_in1", "sd_ic1");
    end_loop(lb1);
    auto *mean =
        g_ir->CreateFDiv(sn1, llvm::ConstantFP::get(dbl, (double)sz), "sd_m");

    auto *entry2 = g_ir->GetInsertBlock();
    auto *loop2 = llvm::BasicBlock::Create(*g_ctx, "std_s2", fn);
    auto *exit2 = llvm::BasicBlock::Create(*g_ctx, "std_e2");
    g_ir->CreateBr(loop2);
    g_ir->SetInsertPoint(loop2);
    auto *i2 = g_ir->CreatePHI(i64, 2, "sd_i2");
    i2->addIncoming(zero_i, entry2);
    auto *s2 = g_ir->CreatePHI(dbl, 2, "sd_s2");
    s2->addIncoming(llvm::ConstantFP::get(dbl, 0.0), entry2);
    auto *v2 =
        g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, xv, i2, "sd_g2"), "sd_v2");
    auto *d = g_ir->CreateFSub(v2, mean, "sd_d");
    auto *sn2 = g_ir->CreateFAdd(s2, g_ir->CreateFMul(d, d, "sd_d2"), "sd_sn2");
    auto *in2 = g_ir->CreateAdd(i2, llvm::ConstantInt::get(i64, 1), "sd_in2");
    i2->addIncoming(in2, loop2);
    s2->addIncoming(sn2, loop2);
    auto *ic2 = g_ir->CreateICmpSLT(in2, sz_c, "sd_ic2");
    g_ir->CreateCondBr(ic2, loop2, exit2);
    fn->insert(fn->end(), exit2);
    g_ir->SetInsertPoint(exit2);
    auto *var = g_ir->CreateFDiv(
        sn2, llvm::ConstantFP::get(dbl, (double)(sz - 1)), "sd_var");
    auto *sqrt_intr = llvm::Intrinsic::getOrInsertDeclaration(
        g_mod.get(), llvm::Intrinsic::sqrt, {dbl});
    return g_ir->CreateCall(sqrt_intr, var, "std");
}

static llvm::Value *
compile_cov_corr(const std::string &callee,
                 const std::vector<std::unique_ptr<Node>> &args)
{
    if (args.size() != 2)
        return err_v((callee + " requires exactly 2 arguments: " + callee +
                      "(array1, array2)")
                         .c_str());
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    int sz = get_size(args[0].get(), callee.c_str());
    if (sz < 0)
        return err_v((callee + ": cannot determine size").c_str());
    auto *xv = args[0]->compile();
    auto *yv = args[1]->compile();
    if (!xv || !yv)
        return nullptr;
    auto *fn = g_ir->GetInsertBlock()->getParent();
    auto *entry = g_ir->GetInsertBlock();
    auto *zero_i = llvm::ConstantInt::get(i64, 0);
    auto *sz_c = llvm::ConstantInt::get(i64, sz);

    auto *loop1 = llvm::BasicBlock::Create(*g_ctx, "cc_s1", fn);
    auto *exit1 = llvm::BasicBlock::Create(*g_ctx, "cc_e1");
    g_ir->CreateBr(loop1);
    g_ir->SetInsertPoint(loop1);
    auto *i1 = g_ir->CreatePHI(i64, 2, "cc_i1");
    i1->addIncoming(zero_i, entry);
    auto *sx1 = g_ir->CreatePHI(dbl, 2, "cc_sx1");
    sx1->addIncoming(llvm::ConstantFP::get(dbl, 0.0), entry);
    auto *sy1 = g_ir->CreatePHI(dbl, 2, "cc_sy1");
    sy1->addIncoming(llvm::ConstantFP::get(dbl, 0.0), entry);
    auto *vx1 =
        g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, xv, i1, "cc_gx1"), "cc_vx1");
    auto *vy1 =
        g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, yv, i1, "cc_gy1"), "cc_vy1");
    auto *sxn1 = g_ir->CreateFAdd(sx1, vx1, "cc_sxn1");
    auto *syn1 = g_ir->CreateFAdd(sy1, vy1, "cc_syn1");
    auto *in1 = g_ir->CreateAdd(i1, llvm::ConstantInt::get(i64, 1), "cc_in1");
    i1->addIncoming(in1, loop1);
    sx1->addIncoming(sxn1, loop1);
    sy1->addIncoming(syn1, loop1);
    auto *ic1 = g_ir->CreateICmpSLT(in1, sz_c, "cc_ic1");
    g_ir->CreateCondBr(ic1, loop1, exit1);
    fn->insert(fn->end(), exit1);
    g_ir->SetInsertPoint(exit1);
    auto *mx =
        g_ir->CreateFDiv(sxn1, llvm::ConstantFP::get(dbl, (double)sz), "cc_mx");
    auto *my =
        g_ir->CreateFDiv(syn1, llvm::ConstantFP::get(dbl, (double)sz), "cc_my");

    auto *loop2 = llvm::BasicBlock::Create(*g_ctx, "cc_s2", fn);
    auto *exit2 = llvm::BasicBlock::Create(*g_ctx, "cc_e2");
    g_ir->CreateBr(loop2);
    g_ir->SetInsertPoint(loop2);
    auto *i2 = g_ir->CreatePHI(i64, 2, "cc_i2");
    i2->addIncoming(zero_i, exit1);
    auto *sc = g_ir->CreatePHI(dbl, 2, "cc_sc");
    sc->addIncoming(llvm::ConstantFP::get(dbl, 0.0), exit1);
    auto *vx2 =
        g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, xv, i2, "cc_gx2"), "cc_vx2");
    auto *vy2 =
        g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, yv, i2, "cc_gy2"), "cc_vy2");
    auto *dx = g_ir->CreateFSub(vx2, mx, "cc_dx");
    auto *dy = g_ir->CreateFSub(vy2, my, "cc_dy");
    auto *scn =
        g_ir->CreateFAdd(sc, g_ir->CreateFMul(dx, dy, "cc_dp"), "cc_scn");
    auto *in2 = g_ir->CreateAdd(i2, llvm::ConstantInt::get(i64, 1), "cc_in2");
    i2->addIncoming(in2, loop2);
    sc->addIncoming(scn, loop2);
    auto *ic2 = g_ir->CreateICmpSLT(in2, sz_c, "cc_ic2");
    g_ir->CreateCondBr(ic2, loop2, exit2);
    fn->insert(fn->end(), exit2);
    g_ir->SetInsertPoint(exit2);
    auto *cov_val = g_ir->CreateFDiv(
        scn, llvm::ConstantFP::get(dbl, (double)(sz - 1)), "cov");

    if (callee == "cov")
        return cov_val;

    auto *loop3 = llvm::BasicBlock::Create(*g_ctx, "cc_sx3", fn);
    auto *exit3 = llvm::BasicBlock::Create(*g_ctx, "cc_ex3");
    g_ir->CreateBr(loop3);
    g_ir->SetInsertPoint(loop3);
    auto *i3 = g_ir->CreatePHI(i64, 2, "cc_ix3");
    i3->addIncoming(zero_i, exit2);
    auto *svx = g_ir->CreatePHI(dbl, 2, "cc_svx");
    svx->addIncoming(llvm::ConstantFP::get(dbl, 0.0), exit2);
    auto *vx3 =
        g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, xv, i3, "cc_gx3"), "cc_vx3");
    auto *dx3 = g_ir->CreateFSub(vx3, mx, "cc_dx3");
    auto *svxn =
        g_ir->CreateFAdd(svx, g_ir->CreateFMul(dx3, dx3, "cc_dx2"), "cc_svxn");
    auto *in3 = g_ir->CreateAdd(i3, llvm::ConstantInt::get(i64, 1), "cc_in3");
    i3->addIncoming(in3, loop3);
    svx->addIncoming(svxn, loop3);
    auto *ic3 = g_ir->CreateICmpSLT(in3, sz_c, "cc_ic3");
    g_ir->CreateCondBr(ic3, loop3, exit3);
    fn->insert(fn->end(), exit3);
    g_ir->SetInsertPoint(exit3);
    auto *varx = g_ir->CreateFDiv(
        svxn, llvm::ConstantFP::get(dbl, (double)(sz - 1)), "cc_varx");

    auto *loop4 = llvm::BasicBlock::Create(*g_ctx, "cc_sy4", fn);
    auto *exit4 = llvm::BasicBlock::Create(*g_ctx, "cc_ey4");
    g_ir->CreateBr(loop4);
    g_ir->SetInsertPoint(loop4);
    auto *i4 = g_ir->CreatePHI(i64, 2, "cc_iy4");
    i4->addIncoming(zero_i, exit3);
    auto *svy = g_ir->CreatePHI(dbl, 2, "cc_svy");
    svy->addIncoming(llvm::ConstantFP::get(dbl, 0.0), exit3);
    auto *vy4 =
        g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, yv, i4, "cc_gy4"), "cc_vy4");
    auto *dy4 = g_ir->CreateFSub(vy4, my, "cc_dy4");
    auto *svyn =
        g_ir->CreateFAdd(svy, g_ir->CreateFMul(dy4, dy4, "cc_dy2"), "cc_svyn");
    auto *in4 = g_ir->CreateAdd(i4, llvm::ConstantInt::get(i64, 1), "cc_in4");
    i4->addIncoming(in4, loop4);
    svy->addIncoming(svyn, loop4);
    auto *ic4 = g_ir->CreateICmpSLT(in4, sz_c, "cc_ic4");
    g_ir->CreateCondBr(ic4, loop4, exit4);
    fn->insert(fn->end(), exit4);
    g_ir->SetInsertPoint(exit4);
    auto *vary = g_ir->CreateFDiv(
        svyn, llvm::ConstantFP::get(dbl, (double)(sz - 1)), "cc_vary");
    auto *sqrt_intr = llvm::Intrinsic::getOrInsertDeclaration(
        g_mod.get(), llvm::Intrinsic::sqrt, {dbl});
    auto *sx = g_ir->CreateCall(sqrt_intr, varx, "cc_sx");
    auto *sy = g_ir->CreateCall(sqrt_intr, vary, "cc_sy");
    return g_ir->CreateFDiv(cov_val, g_ir->CreateFMul(sx, sy, "cc_sxy"),
                            "corr");
}

static llvm::Value *compile_norm(const std::vector<std::unique_ptr<Node>> &args)
{
    if (args.size() != 1)
        return err_v("norm requires exactly 1 argument: norm(array)");
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    int sz = get_size(args[0].get(), "norm");
    if (sz < 0)
        return err_v("norm: cannot determine size; use type annotations");
    auto *xv = args[0]->compile();
    if (!xv)
        return nullptr;
    auto lb = start_loop("n_loop");
    auto *i = loop_index(lb, "n_i");
    auto *s = g_ir->CreatePHI(dbl, 2, "n_s");
    s->addIncoming(llvm::ConstantFP::get(dbl, 0.0), lb.entry);
    auto *v = g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, xv, i, "n_g"), "n_v");
    auto *sn = g_ir->CreateFAdd(s, g_ir->CreateFMul(v, v, "n_sq"), "n_sn");
    s->addIncoming(sn, lb.loop);
    auto *sz_c = llvm::ConstantInt::get(i64, sz);
    loop_next(lb, i, sz_c, "n_in", "n_ic");
    end_loop(lb);
    auto *sqrt_intr = llvm::Intrinsic::getOrInsertDeclaration(
        g_mod.get(), llvm::Intrinsic::sqrt, {dbl});
    return g_ir->CreateCall(sqrt_intr, sn, "norm");
}

static llvm::Value *compile_det(const std::vector<std::unique_ptr<Node>> &args)
{
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto dims = get_node_dims(args[0].get());
    if (dims.size() != 2 || dims[0] <= 0 || dims[0] != dims[1])
        return err_v("det expects a square 2D matrix");
    auto *xv = args[0]->compile();
    if (!xv)
        return nullptr;
    auto *ni = llvm::ConstantInt::get(i64, dims[0]);
    auto *det_fn = g_mod->getFunction("nabla_det");
    if (!det_fn)
    {
        auto *dbl_ty = llvm::Type::getDoubleTy(*g_ctx);
        auto *ft = llvm::FunctionType::get(
            dbl_ty, {llvm::PointerType::get(dbl_ty, 0), i64}, false);
        det_fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                        "nabla_det", *g_mod);
    }
    return g_ir->CreateCall(det_fn, {xv, ni}, "det");
}

static llvm::Value *compile_eig(const std::vector<std::unique_ptr<Node>> &args)
{
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto dims = get_node_dims(args[0].get());
    if (dims.size() != 2 || dims[0] <= 0 || dims[0] != dims[1])
        return err_v("eig expects a square 2D matrix");
    auto *xv = args[0]->compile();
    if (!xv)
        return nullptr;
    auto *ni = llvm::ConstantInt::get(i64, dims[0]);
    auto *eig_fn = g_mod->getFunction("nabla_eig");
    if (!eig_fn)
    {
        auto *dbl_ty = llvm::Type::getDoubleTy(*g_ctx);
        auto *ret_ty = llvm::PointerType::get(dbl_ty, 0);
        auto *ft = llvm::FunctionType::get(
            ret_ty, {llvm::PointerType::get(dbl_ty, 0), i64}, false);
        eig_fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                        "nabla_eig", *g_mod);
    }
    return g_ir->CreateCall(eig_fn, {xv, ni}, "eig");
}

static llvm::Value *
compile_solve(const std::vector<std::unique_ptr<Node>> &args)
{
    if (args.size() != 2)
        return err_v(
            "solve requires exactly 2 arguments: solve(A_matrix, b_vector)");
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto adims = get_node_dims(args[0].get());
    auto bdims = get_node_dims(args[1].get());
    if (adims.size() != 2 || adims[0] <= 0 || adims[0] != adims[1])
        return err_v("solve expects A to be a square 2D matrix");
    if (!(bdims.size() == 1 || (bdims.size() == 2 && bdims[1] == 1)))
        return err_v("solve expects b to be a vector or an n x 1 matrix");
    auto *av = args[0]->compile();
    auto *bv = args[1]->compile();
    if (!av || !bv)
        return nullptr;
    auto *ni = llvm::ConstantInt::get(i64, adims[0]);
    auto *solve_fn = g_mod->getFunction("nabla_solve");
    if (!solve_fn)
    {
        auto *dbl_ty = llvm::Type::getDoubleTy(*g_ctx);
        auto *ret_ty = llvm::PointerType::get(dbl_ty, 0);
        auto *ft =
            llvm::FunctionType::get(ret_ty,
                                    {llvm::PointerType::get(dbl_ty, 0),
                                     llvm::PointerType::get(dbl_ty, 0), i64},
                                    false);
        solve_fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                          "nabla_solve", *g_mod);
    }
    return g_ir->CreateCall(solve_fn, {av, bv, ni}, "solve");
}

static llvm::Value *compile_inv(const std::vector<std::unique_ptr<Node>> &args)
{
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto dims = get_node_dims(args[0].get());
    if (dims.size() != 2 || dims[0] <= 0 || dims[0] != dims[1])
        return err_v("inv expects a square 2D matrix");
    auto *xv = args[0]->compile();
    if (!xv)
        return nullptr;
    auto *ni = llvm::ConstantInt::get(i64, dims[0]);
    auto *inv_fn = g_mod->getFunction("nabla_inv");
    if (!inv_fn)
    {
        auto *dbl_ty = llvm::Type::getDoubleTy(*g_ctx);
        auto *ret_ty = llvm::PointerType::get(dbl_ty, 0);
        auto *ft = llvm::FunctionType::get(
            ret_ty, {llvm::PointerType::get(dbl_ty, 0), i64}, false);
        inv_fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                        "nabla_inv", *g_mod);
    }
    return g_ir->CreateCall(inv_fn, {xv, ni}, "inv");
}

static llvm::Value *compile_len(const std::vector<std::unique_ptr<Node>> &args)
{
    if (args.empty())
        return err_v("len: expected 1 argument");
    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);

    if (auto *arr = dynamic_cast<ArrNode *>(args[0].get()))
        return llvm::ConstantFP::get(dbl, (double)arr->size());

    if (auto *sym = dynamic_cast<SymNode *>(args[0].get()))
    {
        auto lit = g_var_lens.find(sym->get_name());
        if (lit != g_var_lens.end())
            return llvm::ConstantFP::get(dbl, (double)lit->second);

        auto it = g_var_dims.find(sym->get_name());
        if (it != g_var_dims.end())
            return llvm::ConstantFP::get(
                dbl, (double)(it->second[0] * it->second[1]));
        auto vt = g_var_types.find(sym->get_name());
        if (vt != g_var_types.end() && vt->second == TYPE_STRING)
        {
            auto *v = args[0]->compile();
            if (!v)
                return nullptr;
            auto *strlen_fn = g_mod->getFunction("nabla_strlen");
            return g_ir->CreateCall(strlen_fn, v, "len");
        }
    }

    auto *v = args[0]->compile();
    if (!v)
        return nullptr;
    if (v->getType()->isPointerTy())
    {
        if (auto *sym = dynamic_cast<SymNode *>(args[0].get()))
        {
            auto it = g_var_types.find(sym->get_name());
            if (it != g_var_types.end() && is_array_type(it->second))
            {
                auto lit = g_var_lens.find(sym->get_name());
                if (lit != g_var_lens.end())
                    return llvm::ConstantFP::get(dbl, (double)lit->second);
                return err_v("len: array length unknown");
            }
        }
        auto *strlen_fn = g_mod->getFunction("nabla_strlen");
        return g_ir->CreateCall(strlen_fn, v, "len");
    }
    if (v->getType()->isDoubleTy())
    {
        auto *i64_v = g_ir->CreateFPToSI(v, i64, "d2i");
        auto *ptr_v = g_ir->CreateIntToPtr(
            i64_v, llvm::PointerType::get(*g_ctx, 0), "i2p");
        auto *strlen_fn = g_mod->getFunction("nabla_strlen");
        return g_ir->CreateCall(strlen_fn, ptr_v, "len");
    }
    return err_v("len: cannot determine length");
}

llvm::Value *CallNode::compile()
{
    g_di.emit_pos(this);

    if (callee == "sqrt")
        return compile_sqrt(args);
    if (callee == "sin")
        return compile_trig(args, llvm::Intrinsic::sin, "sin");
    if (callee == "cos")
        return compile_trig(args, llvm::Intrinsic::cos, "cos");
    if (callee == "tan")
        return compile_trig(args, llvm::Intrinsic::tan, "tan");
    if (callee == "input")
        return compile_input();
    if (callee == "type")
        return compile_typeof(args);
    if (callee == "print")
        return compile_print(args);
    if (callee == "eye" || callee == "zeros" || callee == "ones")
        return compile_eye_zeros_ones(callee, args);
    if (callee == "linspace")
        return compile_linspace(args);
    if (callee == "range")
        return compile_range(args);
    if (callee == "mean" || callee == "avr")
        return compile_mean(args);
    if (callee == "std")
        return compile_std(args);
    if (callee == "cov" || callee == "corr")
        return compile_cov_corr(callee, args);
    if (callee == "norm")
        return compile_norm(args);
    if (callee == "det")
        return compile_det(args);
    if (callee == "eig")
        return compile_eig(args);
    if (callee == "solve")
        return compile_solve(args);
    if (callee == "inv")
        return compile_inv(args);
    if (callee == "len")
        return compile_len(args);

    auto *cf = find_func(callee);
    if (!cf)
        return err_v("unknown function");
    if (cf->arg_size() != args.size())
        return err_v("argument count mismatch");

    std::vector<llvm::Value *> av;
    for (auto &a : args)
    {
        auto *v = a->compile();
        if (!v)
            return nullptr;
        av.push_back(v);
    }
    return g_ir->CreateCall(cf, av, "call");
}

llvm::Value *IfNode::compile()
{
    g_di.emit_pos(this);
    auto *fn = g_ir->GetInsertBlock()->getParent();

    auto *merge = llvm::BasicBlock::Create(*g_ctx, "ifcont");
    struct Incoming
    {
        llvm::BasicBlock *bb;
        llvm::Value *val;
    };
    std::vector<Incoming> inc;
    llvm::Type *res_ty = nullptr;

    auto merge_result_type = [&](llvm::Value *v) -> bool
    {
        if (!v)
            return false;
        auto *ty = v->getType();
        if (!res_ty)
        {
            res_ty = ty;
            return true;
        }
        if (res_ty == ty)
            return true;
        if (res_ty->isDoubleTy() && ty->isPointerTy())
            return false;
        if (res_ty->isPointerTy() && ty->isDoubleTy())
            return false;
        if (res_ty->isDoubleTy() && ty->isIntegerTy())
            return true;
        if (res_ty->isIntegerTy() && ty->isDoubleTy())
        {
            res_ty = llvm::Type::getDoubleTy(*g_ctx);
            return true;
        }
        if (res_ty->isIntegerTy(64) && ty->isIntegerTy(1))
            return true;
        if (res_ty->isIntegerTy(1) && ty->isIntegerTy(64))
        {
            res_ty = llvm::Type::getInt64Ty(*g_ctx);
            return true;
        }
        return false;
    };

    auto coerce_branch = [&](llvm::Value *v) -> llvm::Value *
    {
        if (!v || !res_ty)
            return v;
        if (v->getType() == res_ty)
            return v;
        if (res_ty->isDoubleTy())
            return promote_to_double(v);
        if (res_ty->isIntegerTy(64) && v->getType()->isIntegerTy(1))
            return g_ir->CreateZExt(v, res_ty, "if_b2i");
        if (res_ty->isIntegerTy(1) && v->getType()->isIntegerTy(64))
            return g_ir->CreateICmpNE(v, llvm::ConstantInt::get(res_ty, 0),
                                      "if_i2b");
        return v;
    };

    auto *cur_else = llvm::BasicBlock::Create(*g_ctx, "if_entry");

    {
        auto *cv = cond->compile();
        if (!cv)
            return nullptr;
        cv = to_bool_cond(cv);
        auto *tb = llvm::BasicBlock::Create(*g_ctx, "then", fn);
        g_ir->CreateCondBr(cv, tb, cur_else);
        g_ir->SetInsertPoint(tb);
        auto *tv = then_branch->compile();
        if (!tv)
            return nullptr;
        if (!merge_result_type(tv))
            return err_v("mismatched if branch types");
        tv = coerce_branch(tv);
        g_ir->CreateBr(merge);
        inc.push_back({g_ir->GetInsertBlock(), tv});
    }

    fn->insert(fn->end(), cur_else);
    g_ir->SetInsertPoint(cur_else);

    for (auto &e : elifs)
    {
        auto *next_bb = llvm::BasicBlock::Create(*g_ctx, "elif_next");
        auto *cv = e.first->compile();
        if (!cv)
            return nullptr;
        cv = to_bool_cond(cv);
        auto *tb = llvm::BasicBlock::Create(*g_ctx, "elif_then", fn);
        g_ir->CreateCondBr(cv, tb, next_bb);
        g_ir->SetInsertPoint(tb);
        auto *tv = e.second->compile();
        if (!tv)
            return nullptr;
        if (!merge_result_type(tv))
            return err_v("mismatched if branch types");
        tv = coerce_branch(tv);
        g_ir->CreateBr(merge);
        inc.push_back({g_ir->GetInsertBlock(), tv});
        fn->insert(fn->end(), next_bb);
        g_ir->SetInsertPoint(next_bb);
    }

    if (else_branch)
    {
        auto *ev = else_branch->compile();
        if (!ev)
            return nullptr;
        if (!merge_result_type(ev))
            return err_v("mismatched if branch types");
        ev = coerce_branch(ev);
        g_ir->CreateBr(merge);
        inc.push_back({g_ir->GetInsertBlock(), ev});
    }
    else
    {
        if (!res_ty)
            res_ty = llvm::Type::getDoubleTy(*g_ctx);
        if (!res_ty->isDoubleTy())
            return err_v("if expression without else must return a float");
        inc.push_back({g_ir->GetInsertBlock(),
                       llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0))});
        g_ir->CreateBr(merge);
    }

    fn->insert(fn->end(), merge);
    g_ir->SetInsertPoint(merge);
    auto *phi = g_ir->CreatePHI(res_ty, inc.size(), "ifphi");
    for (auto &x : inc)
        phi->addIncoming(x.val, x.bb);
    return phi;
}

llvm::Value *WhileNode::compile()
{
    auto *fn = g_ir->GetInsertBlock()->getParent();
    auto *cond_bb = llvm::BasicBlock::Create(*g_ctx, "wcond", fn);
    auto *body_bb = llvm::BasicBlock::Create(*g_ctx, "wbody", fn);
    auto *after_bb = llvm::BasicBlock::Create(*g_ctx, "wend");

    g_ir->CreateBr(cond_bb);

    g_ir->SetInsertPoint(cond_bb);
    auto *cv = cond->compile();
    if (!cv)
        return nullptr;
    cv = to_bool_cond(cv);
    g_ir->CreateCondBr(cv, body_bb, after_bb);

    g_ir->SetInsertPoint(body_bb);

    auto *ob = g_break_bb, *oc = g_cont_bb;
    g_break_bb = after_bb;
    g_cont_bb = cond_bb;

    if (!body->compile())
        return nullptr;

    g_break_bb = ob;
    g_cont_bb = oc;

    g_ir->CreateBr(cond_bb);

    fn->insert(fn->end(), after_bb);
    g_ir->SetInsertPoint(after_bb);
    return llvm::Constant::getNullValue(llvm::Type::getDoubleTy(*g_ctx));
}

llvm::Value *BreakNode::compile()
{
    if (!g_break_bb)
        return err_v("break outside loop");
    g_ir->CreateBr(g_break_bb);
    auto *ub = llvm::BasicBlock::Create(*g_ctx, "after_b",
                                        g_ir->GetInsertBlock()->getParent());
    g_ir->SetInsertPoint(ub);
    return llvm::UndefValue::get(llvm::Type::getDoubleTy(*g_ctx));
}

llvm::Value *ContinueNode::compile()
{
    if (!g_cont_bb)
        return err_v("continue outside loop");
    g_ir->CreateBr(g_cont_bb);
    auto *ub = llvm::BasicBlock::Create(*g_ctx, "after_c",
                                        g_ir->GetInsertBlock()->getParent());
    g_ir->SetInsertPoint(ub);
    return llvm::UndefValue::get(llvm::Type::getDoubleTy(*g_ctx));
}

llvm::Value *ForNode::compile()
{
    auto *fn = g_ir->GetInsertBlock()->getParent();
    auto *slot = entry_slot(fn, var, llvm::Type::getDoubleTy(*g_ctx));
    g_di.emit_pos(this);

    auto *sv = start->compile();
    if (!sv)
        return nullptr;
    g_ir->CreateStore(sv, slot);

    auto *loop_bb = llvm::BasicBlock::Create(*g_ctx, "loop", fn);
    g_ir->CreateBr(loop_bb);
    g_ir->SetInsertPoint(loop_bb);

    auto *prev = g_vars[var];
    g_vars[var] = slot;

    auto *ob = g_break_bb, *oc = g_cont_bb;
    auto *after_bb = llvm::BasicBlock::Create(*g_ctx, "afterloop");
    g_break_bb = after_bb;
    g_cont_bb = loop_bb;

    if (!body->compile())
        return nullptr;

    auto *st = step ? step->compile() : nullptr;
    if (!st)
        st = llvm::ConstantFP::get(*g_ctx, llvm::APFloat(1.0));

    auto *cur =
        g_ir->CreateLoad(llvm::Type::getDoubleTy(*g_ctx), slot, var.c_str());
    g_ir->CreateStore(g_ir->CreateFAdd(cur, st, "next"), slot);

    auto *ev = end->compile();
    if (!ev)
        return nullptr;
    auto *ec = to_bool_cond(ev);
    g_ir->CreateCondBr(ec, loop_bb, after_bb);

    fn->insert(fn->end(), after_bb);
    g_ir->SetInsertPoint(after_bb);

    g_break_bb = ob;
    g_cont_bb = oc;
    if (prev)
        g_vars[var] = prev;
    else
        g_vars.erase(var);

    return llvm::Constant::getNullValue(llvm::Type::getDoubleTy(*g_ctx));
}

llvm::Value *SwitchNode::compile()
{
    auto *fn = g_ir->GetInsertBlock()->getParent();
    auto *val = value->compile();
    if (!val)
        return nullptr;
    auto *merge = llvm::BasicBlock::Create(*g_ctx, "sw_merge");
    struct Inc
    {
        llvm::BasicBlock *bb;
        llvm::Value *val;
    };
    std::vector<Inc> inc;

    for (auto &c : cases)
    {
        auto *cv = c.first->compile();
        if (!cv)
            return nullptr;
        auto *eq = g_ir->CreateFCmpOEQ(val, cv, "c_eq");
        auto *tb = llvm::BasicBlock::Create(*g_ctx, "c_body", fn);
        auto *nb = llvm::BasicBlock::Create(*g_ctx, "c_next");
        g_ir->CreateCondBr(eq, tb, nb);
        g_ir->SetInsertPoint(tb);
        auto *bv = c.second->compile();
        if (!bv)
            return nullptr;
        g_ir->CreateBr(merge);
        inc.push_back({g_ir->GetInsertBlock(), bv});
        fn->insert(fn->end(), nb);
        g_ir->SetInsertPoint(nb);
    }

    if (default_case)
    {
        auto *dv = default_case->compile();
        if (!dv)
            return nullptr;
        g_ir->CreateBr(merge);
        inc.push_back({g_ir->GetInsertBlock(), dv});
    }
    else
    {
        inc.push_back({g_ir->GetInsertBlock(),
                       llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0))});
        g_ir->CreateBr(merge);
    }

    fn->insert(fn->end(), merge);
    g_ir->SetInsertPoint(merge);
    auto *phi =
        g_ir->CreatePHI(llvm::Type::getDoubleTy(*g_ctx), inc.size(), "swphi");
    for (auto &x : inc)
        phi->addIncoming(x.val, x.bb);
    return phi;
}

llvm::Value *LetNode::compile()
{
    std::vector<llvm::AllocaInst *> olds;
    auto *fn = g_ir->GetInsertBlock()->getParent();
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    auto *i64_ty = llvm::Type::getInt64Ty(*g_ctx);
    auto *i1_ty = llvm::Type::getInt1Ty(*g_ctx);

    for (auto &b : bindings)
        if (!b.type && b.init)
            infer_dims(b.init.get(), b.name);

    for (auto &b : bindings)
    {
        llvm::Value *iv = nullptr;
        if (b.init)
        {
            iv = b.init->compile();
            if (!iv)
                return nullptr;
        }

        bool is_matrix = (b.type && b.type->tag == TypeTag::Matrix) ||
                         (get_node_dims(b.init.get()).size() >= 2 &&
                          get_node_dims(b.init.get())[0] > 0);
        bool is_array = b.type && b.type->tag == TypeTag::Array;
        bool is_ptr =
            is_matrix || is_array || (iv && iv->getType()->isPointerTy());
        llvm::Type *var_ty = dbl;
        if (is_ptr)
            var_ty = llvm::PointerType::get(dbl, 0);
        else if (b.type && b.type->tag == TypeTag::Int)
            var_ty = i64_ty;
        else if (b.type && b.type->tag == TypeTag::Bool)
            var_ty = i1_ty;

        auto *slot = entry_slot(fn, b.name, var_ty);

        std::string ltype = TYPE_FLOAT;
        if (b.type && b.type->tag == TypeTag::Int)
            ltype = TYPE_INT;
        else if (b.type && b.type->tag == TypeTag::Bool)
            ltype = TYPE_BOOL;
        else if (b.type && b.type->tag == TypeTag::Str)
            ltype = TYPE_STRING;
        else if (is_array)
            ltype = b.type->to_str();
        else if (is_matrix)
            ltype = b.type ? b.type->to_str() : "matrix";
        else if (iv && iv->getType()->isPointerTy())
        {
            bool is_array_lit =
                dynamic_cast<ArrNode *>(b.init.get()) != nullptr;
            if (!is_array_lit)
                ltype = TYPE_STRING;
            else
                ltype = "[float]";
        }
        else if (iv && iv->getType()->isIntegerTy(1))
            ltype = TYPE_BOOL;
        else if (iv && iv->getType()->isIntegerTy(64))
            ltype = TYPE_INT;
        g_var_types[b.name] = ltype;

        if (is_array || dynamic_cast<ArrNode *>(b.init.get()) ||
            dynamic_cast<SymNode *>(b.init.get()))
        {
            int len = get_indexable_len(b.init.get());
            if (len > 0)
                g_var_lens[b.name] = len;
        }
        if (is_matrix)
        {
            if (b.type && b.type->tag == TypeTag::Matrix)
                g_var_dims[b.name] = b.type->matrix_shape;
            else
                g_var_dims[b.name] = get_node_dims(b.init.get());
        }
        if (iv)
        {
            if (iv->getType() != var_ty)
            {
                if (iv->getType()->isDoubleTy() && var_ty->isIntegerTy(64))
                    iv = g_ir->CreateFPToSI(iv, var_ty, "f2i");
                else if (iv->getType()->isIntegerTy(64) && var_ty->isDoubleTy())
                    iv = g_ir->CreateSIToFP(iv, var_ty, "i2f");
                else if (iv->getType()->isDoubleTy() && var_ty->isIntegerTy(1))
                    iv = g_ir->CreateFCmpONE(
                        iv, llvm::ConstantFP::get(*g_ctx, llvm::APFloat(0.0)),
                        "f2b");
                else if (iv->getType()->isIntegerTy(1) && var_ty->isDoubleTy())
                    iv = g_ir->CreateUIToFP(iv, var_ty, "b2f");
            }
            g_ir->CreateStore(iv, slot);
        }
        else
        {
            llvm::Value *def = llvm::Constant::getNullValue(var_ty);
            g_ir->CreateStore(def, slot);
        }
        olds.push_back(g_vars[b.name]);
        g_vars[b.name] = slot;
    }

    g_di.emit_pos(this);
    auto *bv = body->compile();
    if (!bv)
        return nullptr;

    for (size_t i = 0; i < bindings.size(); ++i)
    {
        if (olds[i])
            g_vars[bindings[i].name] = olds[i];
        else
            g_vars.erase(bindings[i].name);
        g_var_dims.erase(bindings[i].name);
        g_var_lens.erase(bindings[i].name);
        g_var_types.erase(bindings[i].name);
    }
    return bv;
}

llvm::Value *GlobalVarNode::compile()
{
    g_di.emit_pos(this);
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    auto *set_fn = g_mod->getFunction("nabla_set_global");
    auto *decl_fn = g_mod->getFunction("nabla_print_decl");
    auto *str_decl_fn = g_mod->getFunction("nabla_print_str_decl");
    auto *nd_decl_fn = g_mod->getFunction("nabla_print_nd_decl");
    auto *dup_arr_fn = g_mod->getFunction("nabla_dup_array");

    llvm::Value *last = llvm::ConstantFP::get(dbl, 0.0);
    for (auto &b : bindings)
    {
        if (g_var_types.count(b.name))
        {
            std::string msg = "variable '" + b.name + "' already exists";
            return err_v(msg.c_str());
        }
        if (!mutable_)
            g_globals_immutable.insert(b.name);

        llvm::Value *val = last;
        bool is_str = false;
        bool is_ptr_like = false;
        std::string type_str = TYPE_FLOAT;
        if (b.init)
        {
            bool is_array_op = false;
            std::vector<int> array_op_dims;
            if (auto *bin = dynamic_cast<BinaryNode *>(b.init.get()))
            {
                auto op = bin->get_op();
                if ((op == "+" || op == "-" || op == "*" || op == "/") &&
                    dynamic_cast<ArrNode *>(bin->get_lhs()) == nullptr &&
                    dynamic_cast<ArrNode *>(bin->get_rhs()) == nullptr)
                {
                    array_op_dims = get_node_dims(bin->get_lhs());
                    if (!array_op_dims.empty())
                        is_array_op = true;
                }
            }

            val = b.init->compile();
            if (!val)
                return nullptr;

            if (is_array_op && !array_op_dims.empty())
            {
                type_str = "[float]";
                g_var_types[b.name] = type_str;
                g_var_dims[b.name] = array_op_dims;
                int len = dims_to_len(array_op_dims);
                if (len > 0)
                    g_var_lens[b.name] = len;
            }
            else if (val->getType()->isPointerTy())
            {
                bool is_array_lit =
                    dynamic_cast<ArrNode *>(b.init.get()) != nullptr;
                auto init_dims = get_node_dims(b.init.get());
                bool is_matrix_type = b.type && b.type->tag == TypeTag::Matrix;
                if (!is_matrix_type)
                    is_matrix_type = init_dims.size() >= 2;
                bool is_array_type_ = b.type && b.type->tag == TypeTag::Array;
                if (!is_array_type_ && init_dims.size() == 1 &&
                    is_array_lit == false)
                    is_array_type_ = true;
                if (!is_array_lit && !is_matrix_type && !is_array_type_)
                {
                    type_str = TYPE_STRING;
                    is_str = true;
                }
                else
                {
                    is_ptr_like = true;
                    if (is_matrix_type)
                    {
                        type_str = (b.type && b.type->tag == TypeTag::Matrix)
                                       ? b.type->to_str()
                                       : "matrix";
                        if (b.type && b.type->tag == TypeTag::Matrix)
                            g_var_dims[b.name] = b.type->matrix_shape;
                        else
                            g_var_dims[b.name] = init_dims;
                    }
                    else if (is_array_type_)
                        type_str = (b.type && b.type->tag == TypeTag::Array)
                                       ? b.type->to_str()
                                       : "[float]";
                    else
                        type_str = "[float]";
                    int len = get_indexable_len(b.init.get());
                    if (len > 0)
                        g_var_lens[b.name] = len;
                    if (is_array_lit && !init_dims.empty())
                        g_var_dims[b.name] = init_dims;
                    else if (is_array_type_ && !init_dims.empty())
                        g_var_dims[b.name] = init_dims;
                }
            }
            else if (b.type && b.type->tag == TypeTag::Array)
            {
                type_str = b.type->to_str();
                is_ptr_like = true;
            }
            else if (b.type && b.type->tag == TypeTag::Matrix)
            {
                type_str = b.type->to_str();
                is_ptr_like = true;
                g_var_dims[b.name] = b.type->matrix_shape;
            }
            else if (val->getType()->isIntegerTy(1))
                type_str = TYPE_BOOL;
            else if (val->getType()->isIntegerTy(64))
                type_str = TYPE_INT;
        }
        g_var_types[b.name] = type_str;
        g_globals[b.name] = 0.0;
        auto *name = g_ir->CreateGlobalStringPtr(b.name, "gname");

        if (is_str || is_ptr_like)
        {
            auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
            if (!val->getType()->isPointerTy())
                return err_v("expected pointer value for global array/string");
            if (is_ptr_like && !is_str)
            {
                int len = -1;
                auto shape_it = g_var_dims.find(b.name);
                if (shape_it != g_var_dims.end())
                    len = dims_to_len(shape_it->second);
                else if (g_var_lens.count(b.name))
                    len = g_var_lens[b.name];
                else
                    len = get_indexable_len(b.init.get());
                if (len < 0)
                    return err_v("array length unknown for global declaration");
                auto *dup_len = llvm::ConstantInt::get(i64, (uint64_t)len);
                val = g_ir->CreateCall(dup_arr_fn, {val, dup_len}, "dup_arr");
            }
            auto *intptr = g_ir->CreatePtrToInt(val, i64, "ptri");
            auto *as_dbl = g_ir->CreateSIToFP(intptr, dbl, "i2d");
            last = g_ir->CreateCall(set_fn, {name, as_dbl});
            if (is_str)
                g_ir->CreateCall(str_decl_fn, {name, val});
            else
            {
                auto shape = g_var_dims.count(b.name)
                                 ? g_var_dims[b.name]
                                 : get_node_dims(b.init.get());
                if (shape.empty())
                    shape = {g_var_lens.count(b.name)
                                 ? g_var_lens[b.name]
                                 : get_indexable_len(b.init.get())};
                if (shape.empty() || dims_to_len(shape) < 0)
                    return err_v(
                        "matrix dimensions unknown for global declaration");
                auto *dims_ptr = emit_dims_ptr(shape, "gshape");
                auto *nd_decl_name =
                    g_ir->CreateGlobalStringPtr(type_str, "tname");
                g_ir->CreateCall(
                    nd_decl_fn,
                    {name, nd_decl_name, val, dims_ptr,
                     llvm::ConstantInt::get(i64, (uint64_t)shape.size())});
            }
        }
        else
        {
            if (val->getType()->isIntegerTy(64) ||
                val->getType()->isIntegerTy(1))
                val = promote_to_double(val);
            last = g_ir->CreateCall(set_fn, {name, val});
            auto *type_str_g = g_ir->CreateGlobalStringPtr(type_str, "tname");
            g_ir->CreateCall(decl_fn, {name, type_str_g, val});
        }
    }
    return last;
}

llvm::Value *ArrNode::compile()
{
    size_t n = elems.size();
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    auto *arr_ty = llvm::ArrayType::get(dbl, n);
    auto *alloca = g_ir->CreateAlloca(arr_ty, nullptr, "arr");
    for (size_t i = 0; i < n; ++i)
    {
        auto *ev = elems[i]->compile();
        if (!ev)
            return nullptr;
        if (ev->getType()->isIntegerTy())
            ev = g_ir->CreateSIToFP(ev, dbl, "itof");
        auto *gep = g_ir->CreateGEP(
            arr_ty, alloca,
            {llvm::ConstantInt::get(llvm::Type::getInt64Ty(*g_ctx), 0),
             llvm::ConstantInt::get(llvm::Type::getInt64Ty(*g_ctx), i)});
        g_ir->CreateStore(ev, gep);
    }
    return g_ir->CreateGEP(
        arr_ty, alloca,
        {llvm::ConstantInt::get(llvm::Type::getInt64Ty(*g_ctx), 0),
         llvm::ConstantInt::get(llvm::Type::getInt64Ty(*g_ctx), 0)});
}

llvm::Value *IdxNode::compile()
{
    auto *tv = target->compile();
    if (!tv)
        return nullptr;
    if (indices.empty())
        return nullptr;

    bool is_str = false;
    bool is_matrix = false;
    if (auto *sym = dynamic_cast<SymNode *>(target.get()))
    {
        auto it = g_var_types.find(sym->get_name());
        is_str = (it != g_var_types.end() && it->second == TYPE_STRING);
        is_matrix = g_var_dims.count(sym->get_name()) > 0;
        if (is_str && is_matrix)
            is_str = false;
    }
    if (dynamic_cast<StrNode *>(target.get()))
        is_str = true;

    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);

    if (is_str)
    {
        auto *i8 = llvm::Type::getInt8Ty(*g_ctx);
        if (indices.size() != 1)
            return err_v("string indexing takes exactly one index");
        auto *iv = indices[0]->compile();
        if (!iv)
            return nullptr;
        auto *idx_i64 = to_i64_index(iv);
        int static_len = get_indexable_len(target.get());
        if (auto *str = dynamic_cast<StrNode *>(target.get()))
        {
            size_t len = str->value().size();
            if (auto *intn = dynamic_cast<IntNode *>(indices[0].get()))
            {
                int64_t idx_val = intn->value();
                if (idx_val < 0 || (size_t)idx_val >= len)
                {
                    err("string index out of bounds: %lld (length %zu)",
                        (long long)idx_val, len);
                    return nullptr;
                }
            }
            static_len = (int)len;
        }
        if (static_len >= 0)
        {
            auto *len_i64 = llvm::ConstantInt::get(i64, static_len);
            auto *neg = g_ir->CreateICmpSLT(
                idx_i64, llvm::ConstantInt::get(i64, 0), "neg");
            auto *too_big = g_ir->CreateICmpSGE(idx_i64, len_i64, "ob");
            auto *oob = g_ir->CreateOr(neg, too_big, "oob");

            auto *fn = g_ir->GetInsertBlock()->getParent();
            auto *ok_bb = llvm::BasicBlock::Create(*g_ctx, "ok", fn);
            auto *err_bb = llvm::BasicBlock::Create(*g_ctx, "err", fn);
            g_ir->CreateCondBr(oob, err_bb, ok_bb);

            g_ir->SetInsertPoint(err_bb);
            emit_bounds_error("string", idx_i64, len_i64);

            g_ir->SetInsertPoint(ok_bb);
        }
        else
        {
            auto *strlen_fn = g_mod->getFunction("nabla_strlen");
            auto *len_d = g_ir->CreateCall(strlen_fn, {tv}, "slen");
            auto *len_i64 = g_ir->CreateFPToSI(len_d, i64, "slen_i");
            auto *neg = g_ir->CreateICmpSLT(
                idx_i64, llvm::ConstantInt::get(i64, 0), "neg");
            auto *too_big = g_ir->CreateICmpSGE(idx_i64, len_i64, "ob");
            auto *oob = g_ir->CreateOr(neg, too_big, "oob");

            auto *fn = g_ir->GetInsertBlock()->getParent();
            auto *ok_bb = llvm::BasicBlock::Create(*g_ctx, "ok", fn);
            auto *err_bb = llvm::BasicBlock::Create(*g_ctx, "err", fn);
            g_ir->CreateCondBr(oob, err_bb, ok_bb);

            g_ir->SetInsertPoint(err_bb);
            emit_bounds_error("string", idx_i64, len_i64);

            g_ir->SetInsertPoint(ok_bb);
        }
        auto *gep = g_ir->CreateGEP(i8, tv, {idx_i64}, "sidx");
        auto *ch = g_ir->CreateLoad(i8, gep, "ch");
        return g_ir->CreateSExt(ch, i64, "ch_i");
    }

    auto *dbl_ty = llvm::Type::getDoubleTy(*g_ctx);
    auto shape = get_node_dims(target.get());
    if (shape.empty())
        return err_v("indexing target shape unknown");

    std::vector<llvm::Value *> idx_vals;
    idx_vals.reserve(indices.size());
    for (auto &idx : indices)
    {
        auto *iv = idx->compile();
        if (!iv)
            return nullptr;
        idx_vals.push_back(iv);
    }

    auto *offset =
        emit_row_major_offset(shape, idx_vals, is_matrix ? "matrix" : "array");
    if (!offset)
        return nullptr;
    auto *gep = g_ir->CreateGEP(dbl_ty, tv, offset, "idx");
    return g_ir->CreateLoad(dbl_ty, gep, "idxv");
}

llvm::Value *FieldNode::compile()
{
    auto *tv = target->compile();
    if (!tv)
        return nullptr;

    for (auto &[sn, def] : g_struct_types)
    {
        auto *lst = g_llvm_structs[sn];
        if (!lst)
            continue;
        auto *expected = llvm::PointerType::get(lst, 0);
        if (tv->getType() != expected)
            continue;

        for (unsigned i = 0; i < def.size(); ++i)
        {
            if (def[i].first == field)
            {
                auto *gep = g_ir->CreateGEP(
                    lst, tv,
                    {llvm::ConstantInt::get(llvm::Type::getInt64Ty(*g_ctx), 0),
                     llvm::ConstantInt::get(llvm::Type::getInt64Ty(*g_ctx),
                                            i)});
                return g_ir->CreateLoad(lst->getElementType(i), gep,
                                        field.c_str());
            }
        }
    }

    return err_v("field access on non-struct");
}

llvm::Value *StructNode::compile()
{
    auto *lst = g_llvm_structs[name];
    if (!lst)
    {
        auto &def = g_struct_types[name];
        std::vector<llvm::Type *> members;
        for (auto &f : def)
            members.push_back(llvm::Type::getDoubleTy(*g_ctx));
        lst = llvm::StructType::create(*g_ctx, members, name);
        g_llvm_structs[name] = lst;
    }

    auto *alloca = g_ir->CreateAlloca(lst, nullptr, name.c_str());
    auto &def = g_struct_types[name];

    for (size_t i = 0; i < fields.size() && i < def.size(); ++i)
    {
        auto *fv = fields[i].second->compile();
        if (!fv)
            return nullptr;
        auto *gep = g_ir->CreateGEP(
            lst, alloca,
            {llvm::ConstantInt::get(llvm::Type::getInt64Ty(*g_ctx), 0),
             llvm::ConstantInt::get(llvm::Type::getInt64Ty(*g_ctx), i)});
        g_ir->CreateStore(fv, gep);
    }

    return alloca;
}

llvm::Function *ProtoNode::compile()
{
    std::vector<llvm::Type *> ats;
    for (auto &p : params)
        ats.push_back(llvm_type(p.type.get()));
    auto *ft = llvm::FunctionType::get(llvm_type(ret_type.get()), ats, false);
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name,
                                     g_mod.get());
    unsigned i = 0;
    for (auto &arg : f->args())
        arg.setName(params[i++].name);
    return f;
}

llvm::Value *MatMulNode::compile()
{
    g_di.emit_pos(this);
    auto *la = lhs->compile();
    auto *ra = rhs->compile();
    if (!la || !ra)
        return nullptr;

    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    auto *fn = g_ir->GetInsertBlock()->getParent();

    auto lshape = get_node_dims(lhs.get());
    auto rshape = get_node_dims(rhs.get());
    if (lshape.size() != 2 || rshape.size() != 2)
        return err_v("matrix multiplication expects 2D matrices");
    if (lshape[1] != rshape[0])
        return err_v("matrix multiplication dimension mismatch");

    int m = lshape[0], n = lshape[1], p = rshape[1];

    auto *res_ty = llvm::ArrayType::get(dbl, m * p);
    auto *res = g_ir->CreateAlloca(res_ty, nullptr, "mm_res");
    auto *zero = llvm::ConstantInt::get(i64, 0);
    auto *res_base = g_ir->CreateGEP(res_ty, res, {zero, zero});

    auto *entry_bb = g_ir->GetInsertBlock();
    auto *i_loop = llvm::BasicBlock::Create(*g_ctx, "mm_i", fn);
    auto *j_loop = llvm::BasicBlock::Create(*g_ctx, "mm_j", fn);
    auto *k_loop = llvm::BasicBlock::Create(*g_ctx, "mm_k", fn);
    auto *k_exit = llvm::BasicBlock::Create(*g_ctx, "mm_k_exit", fn);
    auto *i_latch = llvm::BasicBlock::Create(*g_ctx, "mm_i_latch", fn);
    auto *exit_bb = llvm::BasicBlock::Create(*g_ctx, "mm_exit");

    g_ir->CreateBr(i_loop);

    g_ir->SetInsertPoint(i_loop);
    auto *i_phi = g_ir->CreatePHI(i64, 2, "mm_i");
    i_phi->addIncoming(zero, entry_bb);
    auto *i_cond =
        g_ir->CreateICmpSLT(i_phi, llvm::ConstantInt::get(i64, m), "mm_ic");
    g_ir->CreateCondBr(i_cond, j_loop, exit_bb);

    g_ir->SetInsertPoint(j_loop);
    auto *j_phi = g_ir->CreatePHI(i64, 2, "mm_j");
    j_phi->addIncoming(zero, i_loop);
    auto *j_cond =
        g_ir->CreateICmpSLT(j_phi, llvm::ConstantInt::get(i64, p), "mm_jc");
    g_ir->CreateCondBr(j_cond, k_loop, i_latch);

    g_ir->SetInsertPoint(k_loop);
    auto *k_phi = g_ir->CreatePHI(i64, 2, "mm_k");
    k_phi->addIncoming(zero, j_loop);
    auto *s_phi = g_ir->CreatePHI(dbl, 2, "mm_s");
    s_phi->addIncoming(llvm::ConstantFP::get(dbl, 0.0), j_loop);

    auto *a_off = g_ir->CreateAdd(
        g_ir->CreateMul(i_phi, llvm::ConstantInt::get(i64, n), "mm_ar"), k_phi,
        "mm_ao");
    auto *a_v = g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, la, a_off, "mm_ag"),
                                 "mm_av");

    auto *b_off = g_ir->CreateAdd(
        g_ir->CreateMul(k_phi, llvm::ConstantInt::get(i64, p), "mm_br"), j_phi,
        "mm_bo");
    auto *b_v = g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, ra, b_off, "mm_bg"),
                                 "mm_bv");

    auto *s_next =
        g_ir->CreateFAdd(s_phi, g_ir->CreateFMul(a_v, b_v, "mm_pr"), "mm_sn");
    auto *k_next =
        g_ir->CreateAdd(k_phi, llvm::ConstantInt::get(i64, 1), "mm_kn");
    auto *k_cond =
        g_ir->CreateICmpSLT(k_next, llvm::ConstantInt::get(i64, n), "mm_kc");
    k_phi->addIncoming(k_next, k_loop);
    s_phi->addIncoming(s_next, k_loop);
    g_ir->CreateCondBr(k_cond, k_loop, k_exit);

    g_ir->SetInsertPoint(k_exit);
    auto *r_off = g_ir->CreateAdd(
        g_ir->CreateMul(i_phi, llvm::ConstantInt::get(i64, p), "mm_rr"), j_phi,
        "mm_ro");
    g_ir->CreateStore(s_next, g_ir->CreateGEP(dbl, res_base, r_off, "mm_rg"));
    auto *j_next =
        g_ir->CreateAdd(j_phi, llvm::ConstantInt::get(i64, 1), "mm_jn");
    j_phi->addIncoming(j_next, k_exit);
    g_ir->CreateBr(j_loop);

    g_ir->SetInsertPoint(i_latch);
    auto *i_next =
        g_ir->CreateAdd(i_phi, llvm::ConstantInt::get(i64, 1), "mm_in");
    i_phi->addIncoming(i_next, i_latch);
    g_ir->CreateBr(i_loop);

    fn->insert(fn->end(), exit_bb);
    g_ir->SetInsertPoint(exit_bb);
    return res_base;
}

llvm::Value *TransposeNode::compile()
{
    g_di.emit_pos(this);
    auto *op = operand->compile();
    if (!op)
        return nullptr;

    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    auto *fn = g_ir->GetInsertBlock()->getParent();

    auto shape = get_node_dims(operand.get());
    if (shape.size() != 2)
        return err_v("transpose expects a 2D matrix");
    int m = shape[0], n = shape[1];

    auto *res_ty = llvm::ArrayType::get(dbl, m * n);
    auto *res = g_ir->CreateAlloca(res_ty, nullptr, "tp_res");
    auto *zero = llvm::ConstantInt::get(i64, 0);
    auto *res_base = g_ir->CreateGEP(res_ty, res, {zero, zero});

    auto *entry_bb = g_ir->GetInsertBlock();
    auto *i_loop = llvm::BasicBlock::Create(*g_ctx, "tp_i", fn);
    auto *j_loop = llvm::BasicBlock::Create(*g_ctx, "tp_j", fn);
    auto *i_latch = llvm::BasicBlock::Create(*g_ctx, "tp_latch", fn);
    auto *exit_bb = llvm::BasicBlock::Create(*g_ctx, "tp_exit");

    g_ir->CreateBr(i_loop);

    g_ir->SetInsertPoint(i_loop);
    auto *i_phi = g_ir->CreatePHI(i64, 2, "tp_i");
    i_phi->addIncoming(zero, entry_bb);
    auto *i_cond =
        g_ir->CreateICmpSLT(i_phi, llvm::ConstantInt::get(i64, m), "tp_ic");
    g_ir->CreateCondBr(i_cond, j_loop, exit_bb);

    g_ir->SetInsertPoint(j_loop);
    auto *j_phi = g_ir->CreatePHI(i64, 2, "tp_j");
    j_phi->addIncoming(zero, i_loop);

    auto *src_off = g_ir->CreateAdd(
        g_ir->CreateMul(i_phi, llvm::ConstantInt::get(i64, n), "tp_sr"), j_phi,
        "tp_so");
    auto *src_v = g_ir->CreateLoad(
        dbl, g_ir->CreateGEP(dbl, op, src_off, "tp_sg"), "tp_sv");

    auto *dst_off = g_ir->CreateAdd(
        g_ir->CreateMul(j_phi, llvm::ConstantInt::get(i64, m), "tp_dr"), i_phi,
        "tp_do");
    g_ir->CreateStore(src_v, g_ir->CreateGEP(dbl, res_base, dst_off, "tp_dg"));

    auto *j_next =
        g_ir->CreateAdd(j_phi, llvm::ConstantInt::get(i64, 1), "tp_jn");
    j_phi->addIncoming(j_next, j_loop);
    auto *j_cond =
        g_ir->CreateICmpSLT(j_next, llvm::ConstantInt::get(i64, n), "tp_jc");
    g_ir->CreateCondBr(j_cond, j_loop, i_latch);

    g_ir->SetInsertPoint(i_latch);
    auto *i_next =
        g_ir->CreateAdd(i_phi, llvm::ConstantInt::get(i64, 1), "tp_in");
    i_phi->addIncoming(i_next, i_latch);
    g_ir->CreateBr(i_loop);

    fn->insert(fn->end(), exit_bb);
    g_ir->SetInsertPoint(exit_bb);
    return res_base;
}

llvm::Value *DotNode::compile()
{
    g_di.emit_pos(this);
    auto *la = lhs->compile();
    auto *ra = rhs->compile();
    if (!la || !ra)
        return nullptr;

    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    auto *fn = g_ir->GetInsertBlock()->getParent();

    if (vec_size < 0)
        return err_v("dot product size unknown; use typed arrays");
    int n = vec_size;

    auto *entry_bb = g_ir->GetInsertBlock();
    auto *loop_bb = llvm::BasicBlock::Create(*g_ctx, "dot_loop", fn);
    auto *exit_bb = llvm::BasicBlock::Create(*g_ctx, "dot_exit");

    g_ir->CreateBr(loop_bb);

    g_ir->SetInsertPoint(loop_bb);
    auto *i_phi = g_ir->CreatePHI(i64, 2, "dot_i");
    i_phi->addIncoming(llvm::ConstantInt::get(i64, 0), entry_bb);
    auto *s_phi = g_ir->CreatePHI(dbl, 2, "dot_s");
    s_phi->addIncoming(llvm::ConstantFP::get(dbl, 0.0), entry_bb);

    auto *a_v = g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, la, i_phi, "dot_ag"),
                                 "dot_av");
    auto *b_v = g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, ra, i_phi, "dot_bg"),
                                 "dot_bv");
    auto *s_next =
        g_ir->CreateFAdd(s_phi, g_ir->CreateFMul(a_v, b_v, "dot_pr"), "dot_sn");
    auto *i_next =
        g_ir->CreateAdd(i_phi, llvm::ConstantInt::get(i64, 1), "dot_in");
    i_phi->addIncoming(i_next, loop_bb);
    s_phi->addIncoming(s_next, loop_bb);
    auto *i_cond =
        g_ir->CreateICmpSLT(i_next, llvm::ConstantInt::get(i64, n), "dot_ic");
    g_ir->CreateCondBr(i_cond, loop_bb, exit_bb);

    fn->insert(fn->end(), exit_bb);
    g_ir->SetInsertPoint(exit_bb);
    return s_next;
}

llvm::Value *GradNode::compile()
{
    g_di.emit_pos(this);
    auto *pt = point->compile();
    if (!pt)
        return nullptr;

    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    auto *fn = g_ir->GetInsertBlock()->getParent();

    if (pt_size < 0)
        return err_v("gradient point size unknown; use typed arrays");

    auto *eps_v = eps ? eps->compile() : nullptr;
    if (!eps_v)
        eps_v = llvm::ConstantFP::get(dbl, 1e-5);

    auto *f = find_func(fname);
    if (!f)
        return err_v(("unknown function '" + fname + "'").c_str());

    auto *res_ty = llvm::ArrayType::get(dbl, pt_size);
    auto *res = g_ir->CreateAlloca(res_ty, nullptr, "grad_res");
    auto *zero = llvm::ConstantInt::get(i64, 0);
    auto *res_base = g_ir->CreateGEP(res_ty, res, {zero, zero});

    auto *two = llvm::ConstantFP::get(dbl, 2.0);
    auto *two_e = g_ir->CreateFMul(two, eps_v, "grad_2e");

    auto *entry_bb = g_ir->GetInsertBlock();
    auto *loop_bb = llvm::BasicBlock::Create(*g_ctx, "grad_loop", fn);
    auto *exit_bb = llvm::BasicBlock::Create(*g_ctx, "grad_exit");

    auto *tmp_ty = llvm::ArrayType::get(dbl, pt_size);
    auto *tmp = g_ir->CreateAlloca(tmp_ty, nullptr, "grad_tmp");
    auto *tmp_base = g_ir->CreateGEP(tmp_ty, tmp, {zero, zero});

    auto *copy_entry = llvm::BasicBlock::Create(*g_ctx, "grad_copy", fn);
    auto *copy_exit = llvm::BasicBlock::Create(*g_ctx, "grad_copy_end");
    g_ir->CreateBr(copy_entry);

    g_ir->SetInsertPoint(copy_entry);
    auto *ci = g_ir->CreatePHI(i64, 2, "grad_ci");
    ci->addIncoming(zero, entry_bb);
    auto *cp = g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, pt, ci, "grad_cpg"),
                                "grad_cpv");
    g_ir->CreateStore(cp, g_ir->CreateGEP(dbl, tmp_base, ci, "grad_csg"));
    auto *cn = g_ir->CreateAdd(ci, llvm::ConstantInt::get(i64, 1), "grad_cn");
    ci->addIncoming(cn, copy_entry);
    auto *cc = g_ir->CreateICmpSLT(cn, llvm::ConstantInt::get(i64, pt_size),
                                   "grad_cc");
    g_ir->CreateCondBr(cc, copy_entry, copy_exit);

    fn->insert(fn->end(), copy_exit);
    g_ir->SetInsertPoint(copy_exit);
    g_ir->CreateBr(loop_bb);

    g_ir->SetInsertPoint(loop_bb);
    auto *i_phi = g_ir->CreatePHI(i64, 2, "grad_i");
    i_phi->addIncoming(zero, copy_exit);

    auto *tg_off = g_ir->CreateGEP(dbl, tmp_base, i_phi, "grad_tg");
    auto *tv_old = g_ir->CreateLoad(dbl, tg_off, "grad_tv");
    auto *tv_plus = g_ir->CreateFAdd(tv_old, eps_v, "grad_tp");
    g_ir->CreateStore(tv_plus, tg_off);

    auto *f_plus = g_ir->CreateCall(f, tmp_base, "grad_fp");

    auto *tv_minus = g_ir->CreateFSub(tv_plus, two_e, "grad_tm");
    g_ir->CreateStore(tv_minus, tg_off);

    auto *f_minus = g_ir->CreateCall(f, tmp_base, "grad_fm");

    auto *fd = g_ir->CreateFSub(f_plus, f_minus, "grad_fd");
    auto *gr = g_ir->CreateFDiv(fd, two_e, "grad_gr");
    g_ir->CreateStore(gr, g_ir->CreateGEP(dbl, res_base, i_phi, "grad_rg"));

    g_ir->CreateStore(tv_old, tg_off);

    auto *i_next =
        g_ir->CreateAdd(i_phi, llvm::ConstantInt::get(i64, 1), "grad_in");
    i_phi->addIncoming(i_next, loop_bb);
    auto *i_cond = g_ir->CreateICmpSLT(
        i_next, llvm::ConstantInt::get(i64, pt_size), "grad_ic");
    g_ir->CreateCondBr(i_cond, loop_bb, exit_bb);

    fn->insert(fn->end(), exit_bb);
    g_ir->SetInsertPoint(exit_bb);
    return res_base;
}

llvm::Value *JacobianNode::compile()
{
    g_di.emit_pos(this);
    auto *pt = point->compile();
    if (!pt)
        return nullptr;

    auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    auto *fn = g_ir->GetInsertBlock()->getParent();

    if (pt_size < 0)
        return err_v("jacobian point size unknown; use typed arrays");

    auto *eps_v = eps ? eps->compile() : nullptr;
    if (!eps_v)
        eps_v = llvm::ConstantFP::get(dbl, 1e-5);

    auto *f = find_func(fname);
    if (!f)
        return err_v(("unknown function '" + fname + "'").c_str());

    int out_size = get_rows();
    if (out_size <= 0)
        return err_v(
            "jacobian output size unknown; annotate the function return type");

    auto *res_ty = llvm::ArrayType::get(dbl, out_size * pt_size);
    auto *res = g_ir->CreateAlloca(res_ty, nullptr, "jac_res");
    auto *zero = llvm::ConstantInt::get(i64, 0);
    auto *res_base = g_ir->CreateGEP(res_ty, res, {zero, zero});

    auto *two = llvm::ConstantFP::get(dbl, 2.0);
    auto *two_e = g_ir->CreateFMul(two, eps_v, "jac_2e");

    auto *entry_bb = g_ir->GetInsertBlock();
    auto *loop_bb = llvm::BasicBlock::Create(*g_ctx, "jac_loop", fn);
    auto *exit_bb = llvm::BasicBlock::Create(*g_ctx, "jac_exit");

    auto *tmp_ty = llvm::ArrayType::get(dbl, pt_size);
    auto *tmp = g_ir->CreateAlloca(tmp_ty, nullptr, "jac_tmp");
    auto *tmp_base = g_ir->CreateGEP(tmp_ty, tmp, {zero, zero});
    bool ret_is_ptr = f->getReturnType()->isPointerTy();

    auto *copy_entry = llvm::BasicBlock::Create(*g_ctx, "jac_copy", fn);
    auto *copy_exit = llvm::BasicBlock::Create(*g_ctx, "jac_copy_end");
    g_ir->CreateBr(copy_entry);

    g_ir->SetInsertPoint(copy_entry);
    auto *ci = g_ir->CreatePHI(i64, 2, "jac_ci");
    ci->addIncoming(zero, entry_bb);
    auto *cp = g_ir->CreateLoad(dbl, g_ir->CreateGEP(dbl, pt, ci, "jac_cpg"),
                                "jac_cpv");
    g_ir->CreateStore(cp, g_ir->CreateGEP(dbl, tmp_base, ci, "jac_csg"));
    auto *cn = g_ir->CreateAdd(ci, llvm::ConstantInt::get(i64, 1), "jac_cn");
    ci->addIncoming(cn, copy_entry);
    auto *cc =
        g_ir->CreateICmpSLT(cn, llvm::ConstantInt::get(i64, pt_size), "jac_cc");
    g_ir->CreateCondBr(cc, copy_entry, copy_exit);

    fn->insert(fn->end(), copy_exit);
    g_ir->SetInsertPoint(copy_exit);
    g_ir->CreateBr(loop_bb);

    g_ir->SetInsertPoint(loop_bb);
    auto *i_phi = g_ir->CreatePHI(i64, 2, "jac_i");
    i_phi->addIncoming(zero, copy_exit);

    auto *tg_off = g_ir->CreateGEP(dbl, tmp_base, i_phi, "jac_tg");
    auto *tv_old = g_ir->CreateLoad(dbl, tg_off, "jac_tv");
    auto *tv_plus = g_ir->CreateFAdd(tv_old, eps_v, "jac_tp");
    g_ir->CreateStore(tv_plus, tg_off);

    auto *fp_val = g_ir->CreateCall(f, tmp_base, "jac_fp");

    auto *tv_minus = g_ir->CreateFSub(tv_plus, two_e, "jac_tm");
    g_ir->CreateStore(tv_minus, tg_off);

    auto *fm_val = g_ir->CreateCall(f, tmp_base, "jac_fm");

    for (int j = 0; j < out_size; ++j)
    {
        llvm::Value *fp_comp = fp_val;
        llvm::Value *fm_comp = fm_val;
        if (ret_is_ptr)
        {
            auto *j_idx = llvm::ConstantInt::get(i64, j);
            fp_comp = g_ir->CreateLoad(
                dbl, g_ir->CreateGEP(dbl, fp_val, j_idx, "jac_fpg"), "jac_fpv");
            fm_comp = g_ir->CreateLoad(
                dbl, g_ir->CreateGEP(dbl, fm_val, j_idx, "jac_fmg"), "jac_fmv");
        }

        auto *fd = g_ir->CreateFSub(fp_comp, fm_comp, "jac_fd");
        auto *jr = g_ir->CreateFDiv(fd, two_e, "jac_jr");
        auto *out_off = g_ir->CreateAdd(
            g_ir->CreateMul(llvm::ConstantInt::get(i64, j),
                            llvm::ConstantInt::get(i64, pt_size), "jac_or"),
            i_phi, "jac_oi");
        g_ir->CreateStore(jr,
                          g_ir->CreateGEP(dbl, res_base, out_off, "jac_rg"));
    }

    g_ir->CreateStore(tv_old, tg_off);

    auto *i_next =
        g_ir->CreateAdd(i_phi, llvm::ConstantInt::get(i64, 1), "jac_in");
    i_phi->addIncoming(i_next, loop_bb);
    auto *i_cond = g_ir->CreateICmpSLT(
        i_next, llvm::ConstantInt::get(i64, pt_size), "jac_ic");
    g_ir->CreateCondBr(i_cond, loop_bb, exit_bb);

    fn->insert(fn->end(), exit_bb);
    g_ir->SetInsertPoint(exit_bb);
    return res_base;
}

static int dims_to_len(const std::vector<int> &dims)
{
    int len = 1;
    for (int d : dims)
        if (d <= 0)
            return -1;
        else
            len *= d;
    return len;
}

static int call_node_ret_len(const CallNode *call)
{
    const auto &cn = call->get_callee();
    const auto &ca = call->get_args();

    if (cn == "eye")
    {
        int d = ca.empty() ? -1 : get_const_int(ca[0].get());
        return (d > 0) ? d * d : -1;
    }
    if (cn == "zeros" || cn == "ones")
    {
        if (ca.size() >= 2)
        {
            int r = get_const_int(ca[0].get());
            int c = get_const_int(ca[1].get());
            return (r > 0 && c > 0) ? r * c : -1;
        }
        if (!ca.empty())
        {
            int d = get_const_int(ca[0].get());
            return (d > 0) ? d : -1;
        }
        return -1;
    }
    if (cn == "linspace")
    {
        int d = ca.size() < 3 ? -1 : get_const_int(ca[2].get());
        return (d > 0) ? d : -1;
    }
    if (cn == "solve")
    {
        if (ca.size() >= 2)
        {
            auto dims = get_node_dims(ca[1].get());
            if (dims.size() >= 2 && dims[0] > 0 && dims[1] == 1)
                return dims[0];
            else if (dims.size() == 1 && dims[0] > 0)
                return dims[0];
        }
        return -1;
    }
    if (cn == "inv")
    {
        if (!ca.empty())
        {
            auto dims = get_node_dims(ca[0].get());
            if (dims.size() >= 2 && dims[0] > 0 && dims[0] == dims[1])
                return dims_to_len(dims);
        }
        return -1;
    }
    if (cn == "eig")
    {
        if (!ca.empty())
        {
            auto dims = get_node_dims(ca[0].get());
            if (dims.size() >= 2 && dims[0] > 0)
                return dims[0];
        }
        return -1;
    }
    return -1;
}

static int get_expr_ret_len(Node *n)
{
    if (!n)
        return -1;
    if (auto *arr = dynamic_cast<ArrNode *>(n))
        return (int)arr->size();
    if (auto *mm = dynamic_cast<MatMulNode *>(n))
        return dims_to_len({mm->get_rows(), mm->get_cols()});
    if (auto *tp = dynamic_cast<TransposeNode *>(n))
        return dims_to_len({tp->get_rows(), tp->get_cols()});
    if (auto *gr = dynamic_cast<GradNode *>(n))
        return dims_to_len({gr->get_rows(), gr->get_cols()});
    if (auto *jac = dynamic_cast<JacobianNode *>(n))
        return dims_to_len({jac->get_rows(), jac->get_cols()});
    if (auto *let = dynamic_cast<LetNode *>(n))
    {
        if (auto *sym = dynamic_cast<SymNode *>(let->get_body()))
        {
            for (auto &b : let->get_bindings())
            {
                if (b.name != sym->get_name())
                    continue;
                if (b.type && b.type->tag == TypeTag::Matrix)
                    return dims_to_len(b.type->matrix_shape);
                if (b.type && b.type->tag == TypeTag::Array && b.init)
                    if (auto *arr = dynamic_cast<ArrNode *>(b.init.get()))
                        return (int)arr->size();
                if (b.init)
                {
                    int len = get_expr_ret_len(b.init.get());
                    if (len > 0)
                        return len;
                }
            }
        }
        return get_expr_ret_len(let->get_body());
    }
    if (auto *sym = dynamic_cast<SymNode *>(n))
    {
        auto lit = g_var_lens.find(sym->get_name());
        if (lit != g_var_lens.end())
            return lit->second;
        auto dim = g_var_dims.find(sym->get_name());
        if (dim != g_var_dims.end())
            return dims_to_len(dim->second);
    }
    if (auto *call = dynamic_cast<CallNode *>(n))
        return call_node_ret_len(call);
    return -1;
}

llvm::Function *FuncNode::compile()
{
    auto &hdr = *proto;
    g_protos[proto->get_name()] = std::move(proto);
    auto *f = find_func(hdr.get_name());
    if (!f)
        return nullptr;

    if (hdr.is_binary())
        g_op_prec[std::string(1, hdr.operator_char())] = hdr.operator_prec();

    auto *bb = llvm::BasicBlock::Create(*g_ctx, "entry", f);
    g_ir->SetInsertPoint(bb);

    auto *unit =
        g_dbg->createFile(g_di.cu->getFilename(), g_di.cu->getDirectory());
    auto *sp = g_dbg->createFunction(
        unit, hdr.get_name(), llvm::StringRef(), unit, hdr.get_line(),
        mk_func_ty(f->arg_size()), hdr.get_line(), llvm::DINode::FlagPrototyped,
        llvm::DISubprogram::SPFlagDefinition);
    f->setSubprogram(sp);
    g_di.scopes.push_back(sp);
    g_di.emit_pos(nullptr);

    g_vars.clear();
    std::vector<std::pair<std::string, std::string>> old_types;
    std::vector<std::pair<std::string, int>> old_lens;
    std::vector<std::pair<std::string, std::vector<int>>> old_dims;
    auto restore_param_state = [&]()
    {
        for (auto &item : old_types)
        {
            if (item.second.empty())
                g_var_types.erase(item.first);
            else
                g_var_types[item.first] = item.second;
        }
        for (auto &item : old_lens)
        {
            if (item.second == 0)
                g_var_lens.erase(item.first);
            else
                g_var_lens[item.first] = item.second;
        }
        for (auto &item : old_dims)
        {
            if (!item.second.empty() && item.second[0] == -1)
                g_var_dims.erase(item.first);
            else
                g_var_dims[item.first] = item.second;
        }
    };
    unsigned aidx = 0;
    for (auto &arg : f->args())
    {
        auto *at = llvm_type(hdr.get_params()[aidx].type.get());
        auto *slot = entry_slot(f, arg.getName(), at);
        const auto &param = hdr.get_params()[aidx];
        old_types.push_back({std::string(arg.getName()),
                             g_var_types.count(std::string(arg.getName()))
                                 ? g_var_types[std::string(arg.getName())]
                                 : ""});
        old_lens.push_back({std::string(arg.getName()),
                            g_var_lens.count(std::string(arg.getName()))
                                ? g_var_lens[std::string(arg.getName())]
                                : 0});
        auto dit = g_var_dims.find(std::string(arg.getName()));
        if (dit != g_var_dims.end())
            old_dims.push_back({std::string(arg.getName()), dit->second});
        else
            old_dims.push_back({std::string(arg.getName()), {-1}});

        g_var_types[std::string(arg.getName())] =
            param.type ? param.type->to_str() : TYPE_FLOAT;
        if (param.type && param.type->tag == TypeTag::Matrix)
            g_var_dims[std::string(arg.getName())] = param.type->matrix_shape;
        auto *dv = g_dbg->createParameterVariable(sp, arg.getName(), ++aidx,
                                                  unit, hdr.get_line(),
                                                  g_di.dbl_ref(), true);
        g_dbg->insertDeclare(
            slot, dv, g_dbg->createExpression(),
            llvm::DILocation::get(sp->getContext(), hdr.get_line(), 0, sp),
            g_ir->GetInsertBlock());
        g_ir->CreateStore(&arg, slot);
        g_vars[std::string(arg.getName())] = slot;
    }

    g_di.emit_pos(body.get());

    auto *rv = body->compile();
    if (rv)
    {
        bool ret_is_ptr_like =
            hdr.get_ret() && is_pointer_like_type(hdr.get_ret()->to_str());
        if (rv->getType()->isPointerTy())
        {
            int ret_len = get_expr_ret_len(body.get());
            if (ret_len > 0)
            {
                auto *dup_arr_fn = g_mod->getFunction("nabla_dup_array");
                auto *i64 = llvm::Type::getInt64Ty(*g_ctx);
                auto *dup_len = llvm::ConstantInt::get(i64, (uint64_t)ret_len);
                auto *dbl_ptr =
                    llvm::PointerType::get(llvm::Type::getDoubleTy(*g_ctx), 0);
                auto *casted_rv = g_ir->CreateBitCast(rv, dbl_ptr, "rvcast");
                rv = g_ir->CreateCall(dup_arr_fn, {casted_rv, dup_len},
                                      "dup_ret");
            }
            else if (ret_is_ptr_like)
            {
                err("pointer return size unknown; use typed arrays");
                return nullptr;
            }
        }
        if (!ret_is_ptr_like &&
            rv->getType() != llvm::Type::getDoubleTy(*g_ctx))
            rv = promote_to_double(rv);
        g_ir->CreateRet(rv);
        g_di.scopes.pop_back();
        restore_param_state();
        llvm::verifyFunction(*f);
        return f;
    }

    f->eraseFromParent();
    if (hdr.is_binary())
        g_op_prec.erase(std::string(1, hdr.operator_char()));
    g_di.scopes.pop_back();
    restore_param_state();
    return nullptr;
}
