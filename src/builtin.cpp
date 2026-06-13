#include "builtin.h"
#include "codegen.h"
#include "constants.h"
#include "context.h"
#include "export.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include <csetjmp>
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" DLLEXPORT double putchard(double x)
{
    fputc((char)x, stderr);
    return 0;
}

extern "C" DLLEXPORT double printd(double x)
{
    fprintf(stderr, "%f", x);
    return 0;
}

extern "C" DLLEXPORT double nabla_print_str(const char *s)
{
    if (s)
        fputs(s, stderr);
    return 0;
}

extern "C" DLLEXPORT double nabla_get_global(const char *name)
{
    auto it = g_globals.find(name);
    if (it != g_globals.end())
        return it->second;
    fprintf(stderr, "Error: unknown variable '%s'\n", name);
    return 0;
}

extern "C" DLLEXPORT double nabla_set_global(const char *name, double val)
{
    g_globals[name] = val;
    return val;
}

extern "C" DLLEXPORT char *nabla_input()
{
    std::string line;
    int ch;
    while ((ch = getchar()) != EOF && ch != '\n' && ch != '\r')
        line += (char)ch;
    if (ch == EOF && line.empty())
        return nullptr;
    char *result = strdup(line.c_str());
    return result;
}

extern "C" DLLEXPORT double nabla_print_decl(const char *name, const char *type,
                                             double val)
{
    if (strcmp(type, TYPE_INT) == 0)
        fprintf(stderr, "%s: %s = %lld\n", name, type, (long long)val);
    else if (strcmp(type, TYPE_BOOL) == 0)
        fprintf(stderr, "%s: %s = %s\n", name, type,
                val != 0.0 ? "true" : "false");
    else
        fprintf(stderr, "%s: %s = %f\n", name, type, val);
    return 0;
}

extern "C" DLLEXPORT double nabla_print_str_decl(const char *name,
                                                 const char *s)
{
    fprintf(stderr, "%s: string = %s\n", name, s ? s : "(null)");
    return 0;
}

extern "C" DLLEXPORT double *nabla_dup_array(const double *data, int64_t len)
{
    if (!data || len <= 0)
        return nullptr;
    auto *result = (double *)malloc(sizeof(double) * (size_t)len);
    if (!result)
        return nullptr;
    memcpy(result, data, sizeof(double) * (size_t)len);
    return result;
}

static int64_t shape_len(const int64_t *dims, int64_t ndims)
{
    if (!dims || ndims <= 0)
        return -1;
    int64_t len = 1;
    for (int64_t i = 0; i < ndims; ++i)
    {
        if (dims[i] <= 0)
            return -1;
        len *= dims[i];
    }
    return len;
}

static int64_t suffix_len(const int64_t *dims, int64_t ndims, int64_t start)
{
    int64_t len = 1;
    for (int64_t i = start; i < ndims; ++i)
        len *= dims[i];
    return len;
}

static void print_nd_value(const double *data, const int64_t *dims,
                           int64_t ndims)
{
    if (!data || !dims || ndims <= 0)
    {
        fprintf(stderr, "[]");
        return;
    }

    if (ndims == 1)
    {
        fprintf(stderr, "[");
        for (int64_t i = 0; i < dims[0]; ++i)
        {
            if (i > 0)
                fprintf(stderr, ", ");
            fprintf(stderr, "%f", data[i]);
        }
        fprintf(stderr, "]");
        return;
    }

    fprintf(stderr, "[");
    auto inner = suffix_len(dims, ndims, 1);
    for (int64_t i = 0; i < dims[0]; ++i)
    {
        if (i > 0)
            fprintf(stderr, "\n");
        fprintf(stderr, " ");
        print_nd_value(data + i * inner, dims + 1, ndims - 1);
    }
    fprintf(stderr, "]");
}

static void print_decl_matrix_like(const char *name, const char *type,
                                   const double *data, const int64_t *dims,
                                   int64_t ndims)
{
    fprintf(stderr, "%s: %s = ", name, type ? type : "array");
    if (!data || !dims || ndims <= 0 || shape_len(dims, ndims) <= 0)
    {
        fprintf(stderr, "[]\n");
        return;
    }
    print_nd_value(data, dims, ndims);
    fprintf(stderr, "\n");
}

extern "C" DLLEXPORT double nabla_print_arr_decl(const char *name,
                                                 const char *type,
                                                 const double *data,
                                                 double rows, double cols)
{
    int64_t dims[2] = {(int64_t)rows, (int64_t)cols};
    print_decl_matrix_like(name, type, data, dims, 2);
    return 0;
}

extern "C" DLLEXPORT double
nabla_print_nd_decl(const char *name, const char *type, const double *data,
                    const int64_t *dims, int64_t ndims)
{
    print_decl_matrix_like(name, type, data, dims, ndims);
    return 0;
}

extern "C" DLLEXPORT double
nabla_print_ndmatrix(const double *data, const int64_t *dims, int64_t ndims)
{
    if (!data || !dims || ndims <= 0)
    {
        fprintf(stderr, "[]\n");
        return 0;
    }
    print_nd_value(data, dims, ndims);
    fprintf(stderr, "\n");
    return 0;
}

extern "C" DLLEXPORT const char *nabla_strcat(const char *a, const char *b)
{
    if (!a && !b)
        return nullptr;
    if (!a)
        return strdup(b);
    if (!b)
        return strdup(a);
    size_t len = strlen(a) + strlen(b) + 1;
    char *result = (char *)malloc(len);
    strcpy(result, a);
    strcat(result, b);
    return result;
}

extern "C" DLLEXPORT double nabla_strlen(const char *s)
{
    if (!s)
        return 0;
    return (double)strlen(s);
}

extern "C" DLLEXPORT void nabla_bounds_error(const char *kind, int64_t idx,
                                             int64_t len)
{
    const char *label = kind ? kind : "index";
    fprintf(stderr, "Error: %s index %lld out of bounds (length %lld)\n", label,
            (long long)idx, (long long)len);
    if (g_repl_mode)
    {
        g_bounds_err = true;
        longjmp(g_bounds_jmp, 1);
    }
    exit(1);
}

extern "C" DLLEXPORT void nabla_div_zero_error()
{
    fprintf(stderr, "Error: division by zero\n");
    if (g_repl_mode)
    {
        g_bounds_err = true;
        longjmp(g_bounds_jmp, 1);
    }
    exit(1);
}

extern "C" DLLEXPORT double nabla_print_matrix(const double *data, double rows,
                                               double cols)
{
    int64_t dims[2] = {(int64_t)rows, (int64_t)cols};
    return nabla_print_ndmatrix(data, dims, 2);
}

void register_builtins()
{
    auto *dbl = llvm::Type::getDoubleTy(*g_ctx);
    auto *i8ptr = llvm::PointerType::get(*g_ctx, 0);
    auto *void_ty = llvm::Type::getVoidTy(*g_ctx);

    auto *putch_ft = llvm::FunctionType::get(dbl, {dbl}, false);
    llvm::Function::Create(putch_ft, llvm::Function::ExternalLinkage,
                           "putchard", *g_mod);

    auto *printd_ft = llvm::FunctionType::get(dbl, {dbl}, false);
    llvm::Function::Create(printd_ft, llvm::Function::ExternalLinkage, "printd",
                           *g_mod);

    auto *print_str_ft = llvm::FunctionType::get(dbl, {i8ptr}, false);
    llvm::Function::Create(print_str_ft, llvm::Function::ExternalLinkage,
                           "nabla_print_str", *g_mod);

    auto *get_global_ft = llvm::FunctionType::get(dbl, {i8ptr}, false);
    llvm::Function::Create(get_global_ft, llvm::Function::ExternalLinkage,
                           "nabla_get_global", *g_mod);

    auto *set_global_ft = llvm::FunctionType::get(dbl, {i8ptr, dbl}, false);
    llvm::Function::Create(set_global_ft, llvm::Function::ExternalLinkage,
                           "nabla_set_global", *g_mod);

    auto *input_ft = llvm::FunctionType::get(i8ptr, {}, false);
    llvm::Function::Create(input_ft, llvm::Function::ExternalLinkage,
                           "nabla_input", *g_mod);

    auto *print_decl_ft =
        llvm::FunctionType::get(dbl, {i8ptr, i8ptr, dbl}, false);
    llvm::Function::Create(print_decl_ft, llvm::Function::ExternalLinkage,
                           "nabla_print_decl", *g_mod);

    auto *print_str_decl_ft =
        llvm::FunctionType::get(dbl, {i8ptr, i8ptr}, false);
    llvm::Function::Create(print_str_decl_ft, llvm::Function::ExternalLinkage,
                           "nabla_print_str_decl", *g_mod);

    auto *print_arr_decl_ft =
        llvm::FunctionType::get(dbl, {i8ptr, i8ptr, i8ptr, dbl, dbl}, false);
    llvm::Function::Create(print_arr_decl_ft, llvm::Function::ExternalLinkage,
                           "nabla_print_arr_decl", *g_mod);

    auto *i64ptr = llvm::PointerType::get(llvm::Type::getInt64Ty(*g_ctx), 0);
    auto *print_nd_decl_ft =
        llvm::FunctionType::get(dbl,
                                {i8ptr, i8ptr, llvm::PointerType::get(dbl, 0),
                                 i64ptr, llvm::Type::getInt64Ty(*g_ctx)},
                                false);
    llvm::Function::Create(print_nd_decl_ft, llvm::Function::ExternalLinkage,
                           "nabla_print_nd_decl", *g_mod);

    auto *print_nd_ft =
        llvm::FunctionType::get(dbl,
                                {llvm::PointerType::get(dbl, 0), i64ptr,
                                 llvm::Type::getInt64Ty(*g_ctx)},
                                false);
    llvm::Function::Create(print_nd_ft, llvm::Function::ExternalLinkage,
                           "nabla_print_ndmatrix", *g_mod);

    auto *dup_arr_ft = llvm::FunctionType::get(
        llvm::PointerType::get(dbl, 0),
        {llvm::PointerType::get(dbl, 0), llvm::Type::getInt64Ty(*g_ctx)},
        false);
    llvm::Function::Create(dup_arr_ft, llvm::Function::ExternalLinkage,
                           "nabla_dup_array", *g_mod);

    auto *strcat_ft = llvm::FunctionType::get(i8ptr, {i8ptr, i8ptr}, false);
    llvm::Function::Create(strcat_ft, llvm::Function::ExternalLinkage,
                           "nabla_strcat", *g_mod);

    auto *strlen_ft = llvm::FunctionType::get(dbl, {i8ptr}, false);
    llvm::Function::Create(strlen_ft, llvm::Function::ExternalLinkage,
                           "nabla_strlen", *g_mod);

    auto *print_mat_ft = llvm::FunctionType::get(dbl, {i8ptr, dbl, dbl}, false);
    llvm::Function::Create(print_mat_ft, llvm::Function::ExternalLinkage,
                           "nabla_print_matrix", *g_mod);

    auto *berr_ft = llvm::FunctionType::get(
        void_ty,
        {i8ptr, llvm::Type::getInt64Ty(*g_ctx), llvm::Type::getInt64Ty(*g_ctx)},
        false);
    llvm::Function::Create(berr_ft, llvm::Function::ExternalLinkage,
                           "nabla_bounds_error", *g_mod);

    auto *dzer_ft = llvm::FunctionType::get(void_ty, {}, false);
    llvm::Function::Create(dzer_ft, llvm::Function::ExternalLinkage,
                           "nabla_div_zero_error", *g_mod);
}
