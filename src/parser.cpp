#include "parser.h"
#include "context.h"
#include "diag.h"
#include <cstdio>
#include <map>

int g_tok;
bool g_parsed_global_decl = false;

static std::map<std::string, std::vector<int>> s_var_dims;

static int infer_point_size(Node *pt)
{
    if (!pt)
        return -1;
    if (auto *arr = dynamic_cast<ArrNode *>(pt))
        return (int)arr->size();
    if (auto *sym = dynamic_cast<SymNode *>(pt))
    {
        auto it = s_var_dims.find(sym->get_name());
        if (it != s_var_dims.end())
        {
            int pt_sz = 1;
            for (int d : it->second)
            {
                if (d <= 0)
                    return -1;
                pt_sz *= d;
            }
            return pt_sz;
        }
    }
    return -1;
}

static int infer_jacobian_output_size(const std::string &fname)
{
    auto it = g_protos.find(fname);
    if (it == g_protos.end())
        return 1;

    const TypeNode *ret = it->second->get_ret();
    if (!ret)
        return 1;

    if (ret->tag == TypeTag::Matrix)
    {
        if (ret->matrix_shape.empty())
            return -1;
        int size = 1;
        for (int d : ret->matrix_shape)
        {
            if (d <= 0)
                return -1;
            size *= d;
        }
        return size;
    }

    if (ret->tag == TypeTag::Array)
        return -1;

    return 1;
}

int advance() { return g_tok = ::next_token(); }

static void skip_newlines()
{
    while (g_tok == '\n')
        advance();
}

static int prec_lookup(const std::string &op)
{
    auto it = g_op_prec.find(op);
    return it != g_op_prec.end() ? it->second : -1;
}

static int tok_prec()
{
    switch (g_tok)
    {
    case '=':
        return prec_lookup("=");
    case '<':
        return prec_lookup("<");
    case '>':
        return prec_lookup(">");
    case '+':
        return prec_lookup("+");
    case '-':
        return prec_lookup("-");
    case '*':
        return prec_lookup("*");
    case '/':
        return prec_lookup("/");
    case '@':
        return prec_lookup("@");
    case '\'':
        return prec_lookup("'");
    case '[':
        return prec_lookup("[");
    case TOK_EQ:
        return prec_lookup("==");
    case TOK_NE:
        return prec_lookup("!=");
    case TOK_LE:
        return prec_lookup("<=");
    case TOK_GE:
        return prec_lookup(">=");
    case TOK_AND:
        return prec_lookup("&&");
    case TOK_OR:
        return prec_lookup("||");
    case TOK_POW:
        return prec_lookup("**");
    default:
        if (g_tok > 0 && g_tok < 256)
            return prec_lookup(std::string(1, (char)g_tok));
        return -1;
    }
}

static std::string tok_op_str()
{
    switch (g_tok)
    {
    case '=':
        return "=";
    case '<':
        return "<";
    case '>':
        return ">";
    case '+':
        return "+";
    case '-':
        return "-";
    case '*':
        return "*";
    case '/':
        return "/";
    case '@':
        return "@";
    case '\'':
        return "'";
    case '[':
        return "[";
    case TOK_EQ:
        return "==";
    case TOK_NE:
        return "!=";
    case TOK_LE:
        return "<=";
    case TOK_GE:
        return ">=";
    case TOK_AND:
        return "&&";
    case TOK_OR:
        return "||";
    case TOK_POW:
        return "**";
    default:
        if (g_tok > 0 && g_tok < 256)
            return std::string(1, (char)g_tok);
        return "";
    }
}

std::unique_ptr<TypeNode> parse_type()
{
    if (g_tok == TOK_INT)
    {
        advance();
        return TypeNode::make_int();
    }
    if (g_tok == TOK_FLOAT)
    {
        advance();
        return TypeNode::make_float();
    }
    if (g_tok == TOK_SYM)
    {
        if (g_sym == "bool")
        {
            advance();
            return TypeNode::make_bool();
        }
        if (g_sym == "string" || g_sym == "str")
        {
            advance();
            return TypeNode::make_str();
        }
        if (g_sym == "void")
        {
            advance();
            return TypeNode::make_void();
        }
        if (g_sym == "matrix")
        {
            advance();
            std::vector<int> shape;
            if (g_tok == '[')
            {
                advance();
                while (true)
                {
                    if (g_tok != TOK_INT)
                    {
                        err("expected dimension size");
                        return nullptr;
                    }
                    shape.push_back((int)g_int_val);
                    advance();
                    if (g_tok == ']')
                    {
                        advance();
                        break;
                    }
                    if (g_tok != ',')
                    {
                        err("expected ',' or ']'");
                        return nullptr;
                    }
                    advance();
                }
            }
            return TypeNode::make_matrix(shape);
        }
        std::string name = g_sym;
        advance();
        return TypeNode::make_named(name);
    }
    if (g_tok == '[')
    {
        advance();
        auto elem = parse_type();
        if (!elem)
            return nullptr;
        if (g_tok != ']')
        {
            err("expected ']' in array type");
            return nullptr;
        }
        advance();
        return TypeNode::make_array(std::move(elem));
    }
    err("expected type");
    return nullptr;
}

static std::unique_ptr<Node> parse_atom();

static std::unique_ptr<Node> parse_prefix()
{
    skip_newlines();
    if (g_tok == TOK_INT)
    {
        auto n = std::make_unique<IntNode>(g_int_val);
        advance();
        return n;
    }
    if (g_tok == TOK_FLOAT)
    {
        auto n = std::make_unique<NumNode>(g_float_val);
        advance();
        return n;
    }
    if (g_tok == TOK_STR)
    {
        auto n = std::make_unique<StrNode>(g_str_val);
        advance();
        return n;
    }
    if (g_tok == TOK_TRUE)
    {
        advance();
        return std::make_unique<BoolNode>(true);
    }
    if (g_tok == TOK_FALSE)
    {
        advance();
        return std::make_unique<BoolNode>(false);
    }
    if (g_tok == TOK_SYM)
    {
        std::string name = g_sym;
        SourcePos pos = g_pos;
        advance();

        if (g_tok == '{')
        {
            advance();
            std::vector<std::pair<std::string, std::unique_ptr<Node>>> fields;
            while (g_tok != '}')
            {
                if (g_tok != TOK_SYM)
                {
                    err("expected field name in struct literal");
                    return nullptr;
                }
                std::string fn = g_sym;
                advance();
                if (g_tok != '=')
                {
                    err("expected '=' in struct field");
                    return nullptr;
                }
                advance();
                auto fv = parse_expr();
                if (!fv)
                    return nullptr;
                fields.push_back({fn, std::move(fv)});
                if (g_tok == ',')
                    advance();
                else if (g_tok != '}')
                    break;
            }
            if (g_tok != '}')
            {
                err("expected '}' after struct fields");
                return nullptr;
            }
            advance();
            return std::make_unique<StructNode>(name, std::move(fields));
        }

        if (g_tok == '(')
        {
            advance();
            std::vector<std::unique_ptr<Node>> args;
            if (g_tok != ')')
            {
                while (true)
                {
                    skip_newlines();
                    if (g_tok == ')')
                        break;
                    auto arg = parse_expr();
                    if (!arg)
                        return nullptr;
                    args.push_back(std::move(arg));
                    skip_newlines();
                    if (g_tok == ')')
                        break;
                    if (g_tok != ',')
                    {
                        err("expected ')' or ',' in call");
                        return nullptr;
                    }
                    advance();
                }
            }
            advance();
            return std::make_unique<CallNode>(pos, name, std::move(args));
        }
        if (g_tok == '.')
        {
            advance();
            if (g_tok != TOK_SYM)
            {
                err("expected field name after '.'");
                return nullptr;
            }
            std::string field = g_sym;
            advance();
            return std::make_unique<FieldNode>(
                std::make_unique<SymNode>(pos, name), field);
        }
        return std::make_unique<SymNode>(pos, name);
    }
    if (g_tok == '(')
    {
        advance();
        if (g_tok == ')')
        {
            advance();
            return nullptr;
        }
        auto expr = parse_expr();
        if (!expr)
            return nullptr;
        if (g_tok != ')')
        {
            err("expected ')'");
            return nullptr;
        }
        advance();
        return expr;
    }
    if (g_tok == '[')
    {
        advance();
        std::vector<std::unique_ptr<Node>> elems;
        if (g_tok != ']')
        {
            while (true)
            {
                auto e = parse_expr();
                if (!e)
                    return nullptr;
                elems.push_back(std::move(e));
                if (g_tok == ']')
                    break;
                if (g_tok != ',')
                {
                    err("expected ']' or ',' in array");
                    return nullptr;
                }
                advance();
            }
        }
        advance();
        return std::make_unique<ArrNode>(std::move(elems));
    }
    if (g_tok == TOK_IF)
    {
        advance();
        auto cond = parse_expr();
        if (!cond)
            return nullptr;
        if (g_tok != TOK_THEN)
        {
            err("expected 'then'");
            return nullptr;
        }
        advance();
        auto then_branch = parse_expr();
        if (!then_branch)
            return nullptr;

        std::vector<std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>>>
            elifs;
        std::unique_ptr<Node> else_branch;

        while (g_tok == TOK_ELIF)
        {
            advance();
            auto econd = parse_expr();
            if (!econd)
                return nullptr;
            if (g_tok != TOK_THEN)
            {
                err("expected 'then' after elif");
                return nullptr;
            }
            advance();
            auto ebody = parse_expr();
            if (!ebody)
                return nullptr;
            elifs.push_back({std::move(econd), std::move(ebody)});
        }

        if (g_tok == TOK_ELSE)
        {
            advance();
            else_branch = parse_expr();
            if (!else_branch)
                return nullptr;
        }

        return std::make_unique<IfNode>(
            cond->line() ? SourcePos{cond->line(), cond->col()} : g_pos,
            std::move(cond), std::move(then_branch), std::move(elifs),
            std::move(else_branch));
    }
    if (g_tok == TOK_WHILE)
    {
        advance();
        auto cond = parse_expr();
        if (!cond)
            return nullptr;
        auto body = parse_expr();
        if (!body)
            return nullptr;
        return std::make_unique<WhileNode>(std::move(cond), std::move(body));
    }
    if (g_tok == TOK_FOR)
    {
        advance();
        if (g_tok != TOK_SYM)
        {
            err("expected identifier after for");
            return nullptr;
        }
        std::string var = g_sym;
        advance();
        if (g_tok != '=')
        {
            err("expected '=' after for");
            return nullptr;
        }
        advance();
        auto start = parse_expr();
        if (!start)
            return nullptr;
        if (g_tok != ',')
        {
            err("expected ',' after for start");
            return nullptr;
        }
        advance();
        auto end = parse_expr();
        if (!end)
            return nullptr;
        std::unique_ptr<Node> step;
        if (g_tok == ',')
        {
            advance();
            step = parse_expr();
            if (!step)
                return nullptr;
        }
        if (g_tok != TOK_IN)
        {
            err("expected 'in' after for");
            return nullptr;
        }
        advance();
        auto body = parse_expr();
        if (!body)
            return nullptr;
        return std::make_unique<ForNode>(var, std::move(start), std::move(end),
                                         std::move(step), std::move(body));
    }
    if (g_tok == TOK_SWITCH)
    {
        advance();
        auto value = parse_expr();
        if (!value)
            return nullptr;

        std::vector<std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>>>
            cases;
        std::unique_ptr<Node> default_case;

        while (g_tok == TOK_CASE)
        {
            advance();
            auto cval = parse_expr();
            if (!cval)
                return nullptr;
            if (g_tok != TOK_THEN)
            {
                err("expected 'then' after case value");
                return nullptr;
            }
            advance();
            auto cbody = parse_expr();
            if (!cbody)
                return nullptr;
            cases.push_back({std::move(cval), std::move(cbody)});
        }

        if (g_tok == TOK_DEFAULT)
        {
            advance();
            default_case = parse_expr();
            if (!default_case)
                return nullptr;
        }

        return std::make_unique<SwitchNode>(std::move(value), std::move(cases),
                                            std::move(default_case));
    }
    if (g_tok == TOK_VAR || g_tok == TOK_MUT)
    {
        bool is_mutable = (g_tok == TOK_MUT);
        advance();
        std::vector<LetBinding> bindings;
        if (g_tok != TOK_SYM)
        {
            err("expected identifier after var");
            return nullptr;
        }
        while (true)
        {
            std::string name = g_sym;
            advance();
            std::unique_ptr<TypeNode> typ;
            std::unique_ptr<Node> init;
            if (g_tok == ':')
            {
                advance();
                typ = parse_type();
                if (!typ)
                    return nullptr;
                if (g_tok == '=')
                {
                    advance();
                    init = parse_expr();
                    if (!init)
                        return nullptr;
                }
            }
            else if (g_tok == '=')
            {
                advance();
                init = parse_expr();
                if (!init)
                    return nullptr;
            }
            if (typ && typ->tag == TypeTag::Matrix)
                s_var_dims[name] = typ->matrix_shape;
            bindings.push_back({name, std::move(typ), std::move(init)});
            if (g_tok != ',')
                break;
            advance();
            if (g_tok != TOK_SYM)
            {
                err("expected identifier after ','");
                return nullptr;
            }
        }
        if (g_tok != TOK_IN)
        {
            g_parsed_global_decl = true;
            return std::make_unique<GlobalVarNode>(std::move(bindings),
                                                   is_mutable);
        }
        advance();
        auto body = parse_expr();
        if (!body)
            return nullptr;
        return std::make_unique<LetNode>(std::move(bindings), std::move(body));
    }
    if (g_tok == TOK_BREAK)
    {
        advance();
        return std::make_unique<BreakNode>();
    }
    if (g_tok == TOK_CONTINUE)
    {
        advance();
        return std::make_unique<ContinueNode>();
    }
    if (g_tok == TOK_DOT)
    {
        advance();
        if (g_tok != '(')
        {
            err("expected '(' after dot");
            return nullptr;
        }
        advance();
        auto a = parse_expr();
        if (!a)
            return nullptr;
        if (g_tok != ',')
        {
            err("expected ',' in dot");
            return nullptr;
        }
        advance();
        auto b = parse_expr();
        if (!b)
            return nullptr;
        if (g_tok != ')')
        {
            err("expected ')' in dot");
            return nullptr;
        }
        advance();
        int n = -1;
        if (auto *sym = dynamic_cast<SymNode *>(a.get()))
        {
            auto it = s_var_dims.find(sym->get_name());
            if (it != s_var_dims.end())
            {
                n = 1;
                for (int d : it->second)
                {
                    if (d <= 0)
                    {
                        n = -1;
                        break;
                    }
                    n *= d;
                }
            }
        }
        if (n < 0)
        {
            if (auto *sym = dynamic_cast<SymNode *>(b.get()))
            {
                auto it = s_var_dims.find(sym->get_name());
                if (it != s_var_dims.end())
                {
                    n = 1;
                    for (int d : it->second)
                    {
                        if (d <= 0)
                        {
                            n = -1;
                            break;
                        }
                        n *= d;
                    }
                }
            }
        }
        return std::make_unique<DotNode>(g_pos, std::move(a), std::move(b), n);
    }
    if (g_tok == TOK_GRAD)
    {
        advance();
        if (g_tok != '(')
        {
            err("expected '(' after grad");
            return nullptr;
        }
        advance();
        if (g_tok != TOK_SYM)
        {
            err("expected function name in grad");
            return nullptr;
        }
        std::string fname = g_sym;
        advance();
        if (g_tok != ',')
        {
            err("expected ',' in grad");
            return nullptr;
        }
        advance();
        auto pt = parse_expr();
        if (!pt)
            return nullptr;
        std::unique_ptr<Node> eps;
        if (g_tok == ',')
        {
            advance();
            eps = parse_expr();
            if (!eps)
                return nullptr;
        }
        if (g_tok != ')')
        {
            err("expected ')' in grad");
            return nullptr;
        }
        advance();
        int pt_sz = -1;
        if (auto *sym = dynamic_cast<SymNode *>(pt.get()))
        {
            auto it = s_var_dims.find(sym->get_name());
            if (it != s_var_dims.end())
            {
                pt_sz = 1;
                for (int d : it->second)
                {
                    if (d <= 0)
                    {
                        pt_sz = -1;
                        break;
                    }
                    pt_sz *= d;
                }
            }
        }
        return std::make_unique<GradNode>(g_pos, fname, std::move(pt),
                                          std::move(eps), pt_sz);
    }
    if (g_tok == TOK_JACOBIAN)
    {
        advance();
        if (g_tok != '(')
        {
            err("expected '(' after jacobian");
            return nullptr;
        }
        advance();
        if (g_tok != TOK_SYM)
        {
            err("expected function name in jacobian");
            return nullptr;
        }
        std::string fname = g_sym;
        advance();
        if (g_tok != ',')
        {
            err("expected ',' in jacobian");
            return nullptr;
        }
        advance();
        auto pt = parse_expr();
        if (!pt)
            return nullptr;
        std::unique_ptr<Node> eps;
        if (g_tok == ',')
        {
            advance();
            eps = parse_expr();
            if (!eps)
                return nullptr;
        }
        if (g_tok != ')')
        {
            err("expected ')' in jacobian");
            return nullptr;
        }
        advance();
        int pt_sz = infer_point_size(pt.get());
        int out_sz = infer_jacobian_output_size(fname);
        return std::make_unique<JacobianNode>(g_pos, fname, std::move(pt),
                                              std::move(eps), pt_sz, out_sz);
    }
    if (g_tok == TOK_NABLA)
    {
        advance();
        if (g_tok != TOK_SYM)
        {
            err("expected function name after ∇");
            return nullptr;
        }
        std::string fname = g_sym;
        SourcePos pos = g_pos;
        advance();
        if (g_tok != '(')
        {
            err("expected '(' after function name");
            return nullptr;
        }
        advance();
        auto pt = parse_expr();
        if (!pt)
            return nullptr;
        if (g_tok != ')')
        {
            err("expected ')'");
            return nullptr;
        }
        advance();
        int pt_sz = infer_point_size(pt.get());
        return std::make_unique<GradNode>(pos, fname, std::move(pt), nullptr,
                                          pt_sz);
    }
    if ((g_tok == '-' || g_tok == '!') || g_tok == TOK_UNA)
    {
        std::string op;
        if (g_tok == '-')
            op = "-";
        else if (g_tok == '!')
            op = "!";
        else
        {
            op = "unary";
            advance();
            if (!isascii(g_tok))
            {
                err("expected unary operator");
                return nullptr;
            }
            op += (char)g_tok;
            advance();
            auto operand = parse_prefix();
            if (!operand)
                return nullptr;
            return std::make_unique<UnaryNode>(op, std::move(operand));
        }
        advance();
        auto operand = parse_prefix();
        if (!operand)
            return nullptr;
        return std::make_unique<UnaryNode>(op, std::move(operand));
    }
    err("unexpected token '%s'", token_desc(g_tok).c_str());
    return nullptr;
}

static std::unique_ptr<Node> parse_infix(int prec, std::unique_ptr<Node> lhs)
{
    while (true)
    {
        int tprec = tok_prec();
        if (tprec < prec)
            return lhs;

        std::string op = tok_op_str();
        SourcePos pos = g_pos;

        if (op == "'")
        {
            advance();
            int nr = -1, nc = -1;
            if (auto *sym = dynamic_cast<SymNode *>(lhs.get()))
            {
                auto it = s_var_dims.find(sym->get_name());
                if (it != s_var_dims.end() && it->second.size() >= 2)
                {
                    nr = it->second[0];
                    nc = it->second[1];
                }
            }
            else if (auto *mm = dynamic_cast<MatMulNode *>(lhs.get()))
            {
                nr = mm->get_rows();
                nc = mm->get_cols();
            }
            else if (auto *tp = dynamic_cast<TransposeNode *>(lhs.get()))
            {
                nr = tp->get_rows();
                nc = tp->get_cols();
            }
            lhs = std::make_unique<TransposeNode>(std::move(lhs), nr, nc);
            continue;
        }

        if (op == "[")
        {
            advance();
            std::vector<std::unique_ptr<Node>> indices;
            while (true)
            {
                auto idx = parse_expr(0);
                if (!idx)
                    return nullptr;
                indices.push_back(std::move(idx));
                if (g_tok == ']')
                {
                    advance();
                    break;
                }
                if (g_tok != ',')
                {
                    err("expected ',' or ']'");
                    return nullptr;
                }
                advance();
            }
            lhs = std::make_unique<IdxNode>(std::move(lhs), std::move(indices));
            continue;
        }

        advance();

        auto rhs = parse_prefix();
        if (!rhs)
            return nullptr;

        int nprec = tok_prec();
        if (tprec < nprec)
        {
            rhs = parse_infix(tprec + 1, std::move(rhs));
            if (!rhs)
                return nullptr;
        }

        if (op == "@")
        {
            int la_r = -1, la_c = -1, lb_c = -1;
            if (auto *sym = dynamic_cast<SymNode *>(lhs.get()))
            {
                auto it = s_var_dims.find(sym->get_name());
                if (it != s_var_dims.end() && it->second.size() >= 2)
                {
                    la_r = it->second[0];
                    la_c = it->second[1];
                }
            }
            else if (auto *mm = dynamic_cast<MatMulNode *>(lhs.get()))
            {
                la_r = mm->get_rows();
                la_c = mm->get_cols();
            }
            else if (auto *tp = dynamic_cast<TransposeNode *>(lhs.get()))
            {
                la_r = tp->get_rows();
                la_c = tp->get_cols();
            }

            if (auto *sym = dynamic_cast<SymNode *>(rhs.get()))
            {
                auto it = s_var_dims.find(sym->get_name());
                if (it != s_var_dims.end() && it->second.size() >= 2)
                    lb_c = it->second[1];
            }
            else if (auto *mm = dynamic_cast<MatMulNode *>(rhs.get()))
                lb_c = mm->get_cols();
            else if (auto *tp = dynamic_cast<TransposeNode *>(rhs.get()))
                lb_c = tp->get_cols();

            lhs = std::make_unique<MatMulNode>(
                pos, std::move(lhs), std::move(rhs), la_r, la_c, lb_c);
        }
        else
            lhs = std::make_unique<BinaryNode>(pos, op, std::move(lhs),
                                               std::move(rhs));
    }
}

std::unique_ptr<Node> parse_expr(int rbp)
{
    skip_newlines();
    auto lhs = parse_prefix();
    if (!lhs)
        return nullptr;
    return parse_infix(rbp, std::move(lhs));
}

std::unique_ptr<ProtoNode> parse_proto()
{
    std::string name;
    SourcePos pos = g_pos;
    unsigned kind = 0;
    unsigned bin_prec = 30;
    auto ret_type = TypeNode::make_float();

    switch (g_tok)
    {
    default:
        err("expected function name in prototype");
        return nullptr;
    case TOK_SYM:
        name = g_sym;
        kind = 0;
        advance();
        break;
    case TOK_UNA:
        advance();
        if (!isascii(g_tok))
        {
            err("expected unary operator");
            return nullptr;
        }
        name = "unary";
        name += (char)g_tok;
        kind = 1;
        advance();
        break;
    case TOK_BIN:
        advance();
        if (!isascii(g_tok))
        {
            err("expected binary operator");
            return nullptr;
        }
        name = "binary";
        name += (char)g_tok;
        kind = 2;
        advance();
        if (g_tok == TOK_INT)
        {
            if (g_int_val < 1 || g_int_val > 100)
            {
                err("invalid precedence %ld", g_int_val);
                return nullptr;
            }
            bin_prec = (unsigned)g_int_val;
            advance();
        }
        break;
    }

    if (g_tok != '(')
    {
        err("expected '(' in prototype");
        return nullptr;
    }
    advance();

    std::vector<Param> params;
    if (g_tok != ')')
    {
        while (true)
        {
            if (g_tok != TOK_SYM)
            {
                err("expected parameter name");
                return nullptr;
            }
            std::string pname = g_sym;
            advance();
            std::unique_ptr<TypeNode> ptype;
            if (g_tok == ':')
            {
                advance();
                ptype = parse_type();
                if (!ptype)
                    return nullptr;
            }
            else
            {
                ptype = TypeNode::make_float();
            }
            params.push_back({pname, std::move(ptype)});
            if (g_tok == ')')
                break;
            if (g_tok != ',')
            {
                err("expected ',' or ')' in params");
                return nullptr;
            }
            advance();
        }
    }

    if (g_tok != ')')
    {
        err("expected ')' in prototype");
        return nullptr;
    }
    advance();

    if (g_tok == TOK_ARROW)
    {
        advance();
        ret_type = parse_type();
        if (!ret_type)
            return nullptr;
    }

    if (kind && params.size() != kind)
    {
        err("invalid number of operands for operator");
        return nullptr;
    }

    return std::make_unique<ProtoNode>(
        pos, name, std::move(params), std::move(ret_type), kind != 0, bin_prec);
}

std::unique_ptr<FuncNode> parse_func()
{
    advance();
    auto proto = parse_proto();
    if (!proto)
        return nullptr;
    auto body = parse_expr();
    if (!body)
        return nullptr;
    return std::make_unique<FuncNode>(std::move(proto), std::move(body));
}

std::unique_ptr<FuncNode> parse_toplevel() { return parse_toplevel_as("main"); }

std::unique_ptr<FuncNode> parse_toplevel_as(const std::string &name)
{
    SourcePos pos = g_pos;
    auto expr = parse_expr();
    if (!expr)
        return nullptr;
    auto proto = std::make_unique<ProtoNode>(pos, name, std::vector<Param>(),
                                             TypeNode::make_float());
    return std::make_unique<FuncNode>(std::move(proto), std::move(expr));
}

std::unique_ptr<ProtoNode> parse_foreign()
{
    advance();
    return parse_proto();
}

std::unique_ptr<StructDef> parse_struct_def()
{
    advance();
    if (g_tok != TOK_SYM)
    {
        err("expected struct name");
        return nullptr;
    }
    std::string name = g_sym;
    advance();
    if (g_tok != '{')
    {
        err("expected '{' after struct name");
        return nullptr;
    }
    advance();
    std::vector<std::pair<std::string, std::unique_ptr<TypeNode>>> fields;
    while (g_tok != '}')
    {
        if (g_tok != TOK_SYM)
        {
            err("expected field name in struct");
            return nullptr;
        }
        std::string fname = g_sym;
        advance();
        if (g_tok != ':')
        {
            err("expected ':' after field name");
            return nullptr;
        }
        advance();
        auto ftype = parse_type();
        if (!ftype)
            return nullptr;
        fields.push_back({fname, std::move(ftype)});
        if (g_tok == ',')
            advance();
    }
    if (g_tok != '}')
    {
        err("expected '}' after struct fields");
        return nullptr;
    }
    advance();
    return std::make_unique<StructDef>(StructDef{name, std::move(fields)});
}
