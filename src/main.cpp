#include "builtin.h"
#include "codegen.h"
#include "constants.h"
#include "context.h"
#include "diag.h"
#include "jit.h"
#include "lexer.h"
#include "parser.h"
#include "token.h"

#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"

#include <algorithm>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <termios.h>
#include <unistd.h>

using namespace llvm;

static ExitOnError g_exit;
static int g_expr_idx;

struct ExprMeta
{
    std::string type;
    int len = -1;
    std::vector<int> dims;
    bool is_decl = false;
};

static std::vector<std::string> g_expr_names;
static std::vector<ExprMeta> g_expr_meta;

static bool is_array_type(const std::string &type)
{
    return !type.empty() && type.front() == '[';
}

static bool is_matrix_type(const std::string &type)
{
    return !type.empty() && type.rfind("matrix", 0) == 0;
}

static void print_array(const double *data, int len)
{
    if (!data || len <= 0)
    {
        fprintf(stderr, "[]\n");
        return;
    }
    fprintf(stderr, "[");
    for (int i = 0; i < len; i++)
    {
        if (i > 0)
            fprintf(stderr, ", ");
        fprintf(stderr, "%f", data[i]);
    }
    fprintf(stderr, "]\n");
}

static int dims_len(const std::vector<int> &dims)
{
    int len = 1;
    for (int d : dims)
    {
        if (d <= 0)
            return -1;
        len *= d;
    }
    return len;
}

static void print_nd_array(const double *data, const std::vector<int> &dims)
{
    if (!data || dims.empty() || dims_len(dims) <= 0)
    {
        fprintf(stderr, "[]\n");
        return;
    }
    if (dims.size() == 1)
    {
        fprintf(stderr, "[");
        for (int i = 0; i < dims[0]; i++)
        {
            if (i > 0)
                fprintf(stderr, ", ");
            fprintf(stderr, "%f", data[i]);
        }
        fprintf(stderr, "]");
        return;
    }

    int inner = 1;
    for (size_t i = 1; i < dims.size(); ++i)
        inner *= dims[i];

    fprintf(stderr, "[");
    for (int i = 0; i < dims[0]; i++)
    {
        if (i > 0)
            fprintf(stderr, "\n");
        fprintf(stderr, " ");
        std::vector<int> tail(dims.begin() + 1, dims.end());
        print_nd_array(data + i * inner, tail);
    }
    fprintf(stderr, "]");
}

static void print_expr_result(double result, const ExprMeta &meta)
{
    const std::string &rtype = meta.type;
    if (rtype == TYPE_BOOL)
    {
        fprintf(stderr, "%s\n", result != 0.0 ? "true" : "false");
    }
    else if (is_matrix_type(rtype))
    {
        auto *arr =
            reinterpret_cast<const double *>(static_cast<int64_t>(result));
        print_nd_array(arr, meta.dims);
        fprintf(stderr, "\n");
        free(const_cast<double *>(arr));
    }
    else if (is_array_type(rtype))
    {
        auto *arr =
            reinterpret_cast<const double *>(static_cast<int64_t>(result));
        if (!meta.dims.empty())
            print_nd_array(arr, meta.dims);
        else
            print_array(arr, meta.len);
        fprintf(stderr, "\n");
        free(const_cast<double *>(arr));
    }
    else if (rtype == TYPE_STRING)
    {
        auto *str =
            reinterpret_cast<const char *>(static_cast<int64_t>(result));
        fprintf(stderr, "%s\n", str ? str : "(null)");
    }
    else if (rtype == TYPE_CHAR)
    {
        fprintf(stderr, "%c\n",
                static_cast<char>(static_cast<int64_t>(result)));
    }
    else
    {
        fprintf(stderr, "%f\n", result);
    }
}

static ExprMeta
infer_expr_meta_impl(Node *body,
                     const std::map<std::string, std::string> &types,
                     const std::map<std::string, int> &lens,
                     const std::map<std::string, std::vector<int>> &dims)
{
    ExprMeta m;
    m.type = TYPE_FLOAT;

    if (auto *sym = dynamic_cast<SymNode *>(body))
    {
        auto it = types.find(sym->get_name());
        if (it != types.end())
            m.type = it->second;
        auto lit = lens.find(sym->get_name());
        if (lit != lens.end())
            m.len = lit->second;
        auto dit = dims.find(sym->get_name());
        if (dit != dims.end())
            m.dims = dit->second;
    }
    else if (dynamic_cast<StrNode *>(body))
    {
        m.type = TYPE_STRING;
    }
    else if (auto *arr = dynamic_cast<ArrNode *>(body))
    {
        m.type = "[float]";
        m.len = (int)arr->size();
        m.dims = {m.len};
    }
    else if (auto *mm = dynamic_cast<MatMulNode *>(body))
    {
        m.type = "matrix";
        m.dims = {mm->get_rows(), mm->get_cols()};
    }
    else if (auto *tp = dynamic_cast<TransposeNode *>(body))
    {
        m.type = "matrix";
        m.dims = {tp->get_rows(), tp->get_cols()};
    }
    else if (auto *gr = dynamic_cast<GradNode *>(body))
    {
        m.type = "matrix";
        m.dims = {gr->get_rows(), gr->get_cols()};
    }
    else if (auto *jac = dynamic_cast<JacobianNode *>(body))
    {
        m.type = "matrix";
        m.dims = {jac->get_rows(), jac->get_cols()};
    }
    else if (auto *let = dynamic_cast<LetNode *>(body))
    {
        auto local_types = types;
        auto local_lens = lens;
        auto local_dims = dims;

        for (const auto &b : let->get_bindings())
        {
            ExprMeta bind_meta;
            if (b.init)
                bind_meta = infer_expr_meta_impl(b.init.get(), local_types,
                                                 local_lens, local_dims);

            if (b.type)
            {
                bind_meta.type = b.type->to_str();
                if (b.type->tag == TypeTag::Array)
                {
                    if (bind_meta.dims.empty() && bind_meta.len > 0)
                        bind_meta.dims = {bind_meta.len};
                }
                else if (b.type && b.type->tag == TypeTag::Matrix)
                {
                    bind_meta.dims = b.type->matrix_shape;
                    bind_meta.len = dims_len(bind_meta.dims);
                }
            }

            if (!bind_meta.type.empty())
                local_types[b.name] = bind_meta.type;
            if (bind_meta.len > 0)
                local_lens[b.name] = bind_meta.len;
            if (!bind_meta.dims.empty())
                local_dims[b.name] = bind_meta.dims;
        }

        return infer_expr_meta_impl(let->get_body(), local_types, local_lens,
                                    local_dims);
    }
    else if (auto *ifn = dynamic_cast<IfNode *>(body))
    {
        auto merge_meta = [](const ExprMeta &lhs, const ExprMeta &rhs)
        {
            ExprMeta out = lhs;
            if (out.type.empty())
                out.type = rhs.type;
            if (out.type != rhs.type)
            {
                bool numeric_lhs = out.type == TYPE_INT ||
                                   out.type == TYPE_FLOAT ||
                                   out.type == TYPE_BOOL;
                bool numeric_rhs = rhs.type == TYPE_INT ||
                                   rhs.type == TYPE_FLOAT ||
                                   rhs.type == TYPE_BOOL;
                if (numeric_lhs && numeric_rhs)
                    out.type = TYPE_FLOAT;
                else
                    out.type.clear();
            }

            if (out.type == rhs.type)
            {
                if (out.len <= 0)
                    out.len = rhs.len;
                if (out.dims.empty())
                    out.dims = rhs.dims;
            }
            return out;
        };

        auto then_meta =
            infer_expr_meta_impl(ifn->get_then_branch(), types, lens, dims);
        ExprMeta result = then_meta;
        for (const auto &elif_branch : ifn->get_elifs())
        {
            auto branch_meta = infer_expr_meta_impl(elif_branch.second.get(),
                                                    types, lens, dims);
            result = merge_meta(result, branch_meta);
        }
        if (ifn->get_else_branch())
        {
            auto else_meta =
                infer_expr_meta_impl(ifn->get_else_branch(), types, lens, dims);
            result = merge_meta(result, else_meta);
        }
        if (result.type.empty())
            result.type = TYPE_FLOAT;
        return result;
    }
    else if (dynamic_cast<BoolNode *>(body))
    {
        m.type = TYPE_BOOL;
    }
    else if (dynamic_cast<IntNode *>(body))
    {
        m.type = TYPE_INT;
    }
    else if (auto *idx = dynamic_cast<IdxNode *>(body))
    {
        auto *target = idx->get_target();
        if (auto *sym = dynamic_cast<SymNode *>(target))
        {
            auto it = types.find(sym->get_name());
            if (it != types.end() && it->second == TYPE_STRING)
                m.type = TYPE_CHAR;
        }
        else if (dynamic_cast<StrNode *>(target))
        {
            m.type = TYPE_CHAR;
        }
    }
    else if (auto *bin = dynamic_cast<BinaryNode *>(body))
    {
        auto lhs_meta = infer_expr_meta_impl(bin->get_lhs(), types, lens, dims);
        auto rhs_meta = infer_expr_meta_impl(bin->get_rhs(), types, lens, dims);

        if (is_array_type(lhs_meta.type) && is_array_type(rhs_meta.type))
        {
            auto op = bin->get_op();
            if (op == "+" || op == "-" || op == "*" || op == "/")
            {
                m.type = lhs_meta.type;
                m.len = lhs_meta.len;
                m.dims = lhs_meta.dims;
            }
        }
    }
    else if (auto *call = dynamic_cast<CallNode *>(body))
    {
        const auto &callee = call->get_callee();
        const auto &args = call->get_args();
        if (callee == "eye" && !args.empty())
        {
            if (auto *in = dynamic_cast<IntNode *>(args[0].get()))
            {
                int d = (int)in->value();
                if (d > 0)
                {
                    m.type = "matrix";
                    m.len = d * d;
                    m.dims = std::vector<int>{d, d};
                }
            }
        }
        else if ((callee == "zeros" || callee == "ones") && !args.empty())
        {
            if (args.size() >= 2)
            {
                auto *rows_node = dynamic_cast<IntNode *>(args[0].get());
                auto *cols_node = dynamic_cast<IntNode *>(args[1].get());
                if (rows_node && cols_node)
                {
                    int rows = (int)rows_node->value();
                    int cols = (int)cols_node->value();
                    if (rows > 0 && cols > 0)
                    {
                        m.type = "matrix";
                        m.len = rows * cols;
                        m.dims = std::vector<int>{rows, cols};
                    }
                }
            }
            else if (auto *in = dynamic_cast<IntNode *>(args[0].get()))
            {
                int d = (int)in->value();
                if (d > 0)
                {
                    m.type = "matrix";
                    m.len = d;
                    m.dims = std::vector<int>{1, d};
                }
            }
        }
        else if (callee == "linspace" && args.size() >= 3)
        {
            if (auto *in = dynamic_cast<IntNode *>(args[2].get()))
            {
                int d = (int)in->value();
                if (d > 0)
                {
                    m.type = "matrix";
                    m.len = d;
                    m.dims = std::vector<int>{1, d};
                }
            }
        }
        else if (callee == "range" && args.size() >= 2)
        {
            auto *start_node = dynamic_cast<IntNode *>(args[0].get());
            auto *end_node = dynamic_cast<IntNode *>(args[1].get());
            if (start_node && end_node)
            {
                int len = (int)end_node->value() - (int)start_node->value() + 1;
                if (len > 0)
                {
                    m.type = "matrix";
                    m.len = len;
                    m.dims = std::vector<int>{1, len};
                }
            }
        }
        else if (callee == "solve" && args.size() >= 2)
        {
            auto b_meta =
                infer_expr_meta_impl(args[1].get(), types, lens, dims);
            if (b_meta.dims.size() == 2 && b_meta.dims[0] > 0 &&
                b_meta.dims[1] == 1)
            {
                m.type = "matrix";
                m.len = b_meta.dims[0];
                m.dims = b_meta.dims;
            }
            else if (b_meta.dims.size() == 1 && b_meta.dims[0] > 0)
            {
                m.type = "matrix";
                m.len = b_meta.dims[0];
                std::vector<int> dims;
                dims.push_back(1);
                dims.push_back(b_meta.dims[0]);
                m.dims = dims;
            }
        }
        else if (callee == "inv" && !args.empty())
        {
            auto a_meta =
                infer_expr_meta_impl(args[0].get(), types, lens, dims);
            if (a_meta.dims.size() == 2 && a_meta.dims[0] > 0 &&
                a_meta.dims[0] == a_meta.dims[1])
            {
                m.type = "matrix";
                m.len = a_meta.dims[0] * a_meta.dims[1];
                m.dims = a_meta.dims;
            }
        }
        else if (callee == "eig" && !args.empty())
        {
            auto a_meta =
                infer_expr_meta_impl(args[0].get(), types, lens, dims);
            if (a_meta.dims.size() == 2 && a_meta.dims[0] > 0 &&
                a_meta.dims[0] == a_meta.dims[1])
            {
                m.type = "matrix";
                m.len = a_meta.dims[0];
                m.dims = std::vector<int>{1, a_meta.dims[0]};
            }
        }
    }
    else if (auto *gvar = dynamic_cast<GlobalVarNode *>(body))
    {
        if (!gvar->get_bindings().empty())
        {
            auto &b = gvar->get_bindings().back();
            auto it = types.find(b.name);
            if (it != types.end())
            {
                m.type = it->second;
                auto len_it = lens.find(b.name);
                if (len_it != lens.end())
                    m.len = len_it->second;
                auto dims_it = dims.find(b.name);
                if (dims_it != dims.end())
                    m.dims = dims_it->second;
            }
        }
    }
    else if (auto *let = dynamic_cast<LetNode *>(body))
    {
        auto local_types = types;
        auto local_lens = lens;
        auto local_dims = dims;

        for (const auto &b : let->get_bindings())
        {
            ExprMeta bind_meta;
            if (b.init)
                bind_meta = infer_expr_meta_impl(b.init.get(), local_types,
                                                 local_lens, local_dims);

            if (b.type)
            {
                bind_meta.type = b.type->to_str();
                if (b.type->tag == TypeTag::Array)
                {
                    if (bind_meta.dims.empty() && bind_meta.len > 0)
                        bind_meta.dims = {bind_meta.len};
                }
                else if (b.type->tag == TypeTag::Matrix)
                {
                    bind_meta.dims = b.type->matrix_shape;
                    bind_meta.len = dims_len(bind_meta.dims);
                }
            }

            if (!bind_meta.type.empty())
                local_types[b.name] = bind_meta.type;
            if (bind_meta.len > 0)
                local_lens[b.name] = bind_meta.len;
            if (!bind_meta.dims.empty())
                local_dims[b.name] = bind_meta.dims;
        }

        return infer_expr_meta_impl(let->get_body(), local_types, local_lens,
                                    local_dims);
    }

    return m;
}

static ExprMeta infer_expr_meta(Node *body)
{
    return infer_expr_meta_impl(body, g_var_types, g_var_lens, g_var_dims);
}

static bool expr_body_is_silent(Node *body)
{
    if (auto *call = dynamic_cast<CallNode *>(body))
        return call->get_callee() == "print" ||
               call->get_callee() == "type" ||
               call->get_callee() == "printd" ||
               call->get_callee() == "putchard";
    if (auto *bin = dynamic_cast<BinaryNode *>(body))
        return bin->get_op() == "=" && dynamic_cast<SymNode *>(bin->get_lhs());
    if (auto *let = dynamic_cast<LetNode *>(body))
        return expr_body_is_silent(let->get_body());
    if (auto *ifn = dynamic_cast<IfNode *>(body))
    {
        if (!ifn->get_else_branch() ||
            !expr_body_is_silent(ifn->get_then_branch()) ||
            !expr_body_is_silent(ifn->get_else_branch()))
            return false;
        for (const auto &elif_branch : ifn->get_elifs())
        {
            if (!expr_body_is_silent(elif_branch.second.get()))
                return false;
        }
        return true;
    }
    return false;
}

static void handle_func()
{
    auto fn = parse_func();
    if (!fn)
    {
        advance();
        return;
    }
    fn->compile();
}

static void handle_foreign()
{
    auto p = parse_foreign();
    if (!p)
    {
        advance();
        return;
    }
    p->compile();
    g_protos[p->get_name()] = std::move(p);
}

static void handle_struct()
{
    auto sd = parse_struct_def();
    if (!sd)
    {
        advance();
        return;
    }
    g_struct_types[sd->name] = std::move(sd->fields);
}

static void handle_expr()
{
    g_parsed_global_decl = false;
    auto name = "__expr_" + std::to_string(g_expr_idx++);
    auto fn = parse_toplevel_as(name);
    if (!fn)
    {
        advance();
        return;
    }
    if (!fn->compile())
        return;

    ExprMeta meta = infer_expr_meta(fn->get_body());
    meta.is_decl = g_parsed_global_decl || expr_body_is_silent(fn->get_body());

    g_expr_names.push_back(name);
    g_expr_meta.push_back(std::move(meta));
}

static void init_module()
{
    setup_mod();
    register_builtins();
    g_mod->addModuleFlag(Module::Warning, "Debug Info Version",
                         DEBUG_METADATA_VERSION);
    if (Triple(sys::getProcessTriple()).isOSDarwin())
        g_mod->addModuleFlag(Module::Warning, "Dwarf Version", 2);
    g_dbg = std::make_unique<DIBuilder>(*g_mod);
    g_di = DiState();
    g_di.cu = g_dbg->createCompileUnit(dwarf::DW_LANG_C,
                                       g_dbg->createFile("fib.ks", "."),
                                       "Nabla", false, "", 0);
}

static void reset_expr_state()
{
    g_expr_names.clear();
    g_expr_meta.clear();
}

static bool run_repl_fn(double (*fn)(), double &result)
{
    g_bounds_err = false;
    if (setjmp(g_bounds_jmp) != 0)
        return false;
    result = fn();
    return true;
}

static void flush_repl()
{
    fflush(stdout);
    g_dbg->finalize();
    auto tsm = orc::ThreadSafeModule(std::move(g_mod), std::move(g_ctx));
    g_exit(g_jit->addModule(std::move(tsm)));

    for (size_t i = 0; i < g_expr_names.size(); i++)
    {
        auto sym = g_exit(g_jit->lookup(g_expr_names[i]));
        auto *fn = sym.getAddress().toPtr<double (*)()>();
        double result = 0.0;
        if (!run_repl_fn(fn, result))
            break;
        if (!g_expr_meta[i].is_decl)
            print_expr_result(result, g_expr_meta[i]);
    }

    reset_expr_state();
    init_module();
}

static int run_jit()
{
    g_dbg->finalize();
    if (has_err())
        return 1;
    auto tsm = orc::ThreadSafeModule(std::move(g_mod), std::move(g_ctx));
    g_exit(g_jit->addModule(std::move(tsm)));
    for (auto &name : g_expr_names)
    {
        auto sym = g_exit(g_jit->lookup(name));
        auto *fn = sym.getAddress().toPtr<double (*)()>();
        fn();
    }
    return 0;
}

static void repl()
{
    while (true)
    {
        switch (g_tok)
        {
        case TOK_EOF:
            return;
        case ';':
            advance();
            break;
        case TOK_DEF:
            handle_func();
            break;
        case TOK_EXT:
            handle_foreign();
            break;
        case TOK_STRUCT:
            handle_struct();
            break;
        default:
            handle_expr();
            break;
        }
    }
}

static std::vector<std::string> s_history;

struct TerminalGuard
{
    struct termios old;
    bool active = false;

    void set()
    {
        if (!isatty(STDIN_FILENO))
            return;
        if (tcgetattr(STDIN_FILENO, &old) != 0)
            return;
        struct termios tio = old;
        tio.c_iflag |= ICRNL;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &tio) == 0)
            active = true;
    }

    ~TerminalGuard()
    {
        if (active)
            tcsetattr(STDIN_FILENO, TCSANOW, &old);
    }
};

static bool read_line(const std::string &prompt, std::string &line)
{
    fprintf(stderr, "%s", prompt.c_str());
    fflush(stderr);
    line.clear();

    int ch;
    while ((ch = getchar()) != EOF && ch != '\n' && ch != '\r')
        line += (char)ch;

    if (ch == EOF && line.empty())
        return false;

    if (!line.empty() && (s_history.empty() || s_history.back() != line))
        s_history.push_back(line);

    return true;
}

static bool unbalanced(const std::string &s)
{
    int depth = 0;
    bool in_str = false;
    for (size_t i = 0; i < s.size(); i++)
    {
        char c = s[i];
        if (c == '"')
        {
            in_str = !in_str;
            continue;
        }
        if (in_str)
            continue;
        if (c == '/' && i + 1 < s.size() && s[i + 1] == '/')
            break;
        switch (c)
        {
        case '{':
        case '(':
        case '[':
            depth++;
            break;
        case '}':
        case ')':
        case ']':
            depth--;
            break;
        }
    }
    return depth > 0;
}

static void dispatch_token()
{
    switch (g_tok)
    {
    case ';':
        advance();
        break;
    case TOK_DEF:
        handle_func();
        break;
    case TOK_EXT:
        handle_foreign();
        break;
    case TOK_STRUCT:
        handle_struct();
        break;
    default:
        handle_expr();
        break;
    }
}

static void run_repl()
{
    TerminalGuard tguard;
    tguard.set();
    g_repl_mode = true;
    g_skip_newlines = false;
    init_module();

    while (true)
    {
        std::string line;
        if (!read_line("> ", line))
            break;
        if (line.empty())
            continue;

        std::string input = line;
        while (unbalanced(input))
        {
            std::string cont;
            if (!read_line("... ", cont))
                goto done;
            if (cont.empty())
                break;
            input += "\n" + cont;
        }

        g_errors.clear();
        set_input_line(input + "\n");
        advance();

        bool errs = false;
        while (g_tok != TOK_EOF && g_tok != '\n')
        {
            dispatch_token();
            if (has_err())
            {
                errs = true;
                break;
            }
        }

        if (errs)
        {
            reset_expr_state();
            continue;
        }

        if (!g_expr_names.empty())
            flush_repl();
    }

done:
    g_repl_mode = false;
    g_dbg->finalize();
}

int main(int argc, char **argv)
{
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

    if (argc > 1 && !freopen(argv[1], "r", stdin))
    {
        fprintf(stderr, "error: cannot open '%s'\n", argv[1]);
        return 1;
    }

    g_prec_init();
    g_jit = g_exit(orc::NablaJIT::Create());

    if (argc <= 1)
    {
        run_repl();
        return 0;
    }

    advance();
    init_module();
    repl();
    return run_jit();
}
