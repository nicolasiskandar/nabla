#include "ast.h"
#include <cstdarg>
#include <cstdio>

static void pad(int indent)
{
    for (int i = 0; i < indent; i++)
        fputs("  ", stderr);
}

static void d(int indent, const char *fmt, ...)
{
    pad(indent);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

void Node::dump(int indent) const { d(indent, "<node>:%d:%d", line(), col()); }

void NumNode::dump(int indent) const { d(indent, "Num:%g", val); }
void IntNode::dump(int indent) const { d(indent, "Int:%ld", val); }
void StrNode::dump(int indent) const { d(indent, "Str:\"%s\"", val.c_str()); }
void BoolNode::dump(int indent) const
{
    d(indent, "Bool:%s", val ? "true" : "false");
}
void SymNode::dump(int indent) const { d(indent, "Sym:%s", name.c_str()); }

void UnaryNode::dump(int indent) const
{
    d(indent, "Unary:%s", op.c_str());
    operand->dump(indent + 1);
}

void BinaryNode::dump(int indent) const
{
    d(indent, "Binary:%s", op.c_str());
    lhs->dump(indent + 1);
    rhs->dump(indent + 1);
}

void CallNode::dump(int indent) const
{
    d(indent, "Call:%s", callee.c_str());
    for (auto &a : args)
        a->dump(indent + 1);
}

void IfNode::dump(int indent) const
{
    d(indent, "If");
    cond->dump(indent + 1);
    then_branch->dump(indent + 1);
    for (auto &e : elifs)
    {
        d(indent, "Elif");
        e.first->dump(indent + 1);
        e.second->dump(indent + 1);
    }
    if (else_branch)
    {
        d(indent, "Else");
        else_branch->dump(indent + 1);
    }
}

void WhileNode::dump(int indent) const
{
    d(indent, "While");
    cond->dump(indent + 1);
    body->dump(indent + 1);
}

void BreakNode::dump(int indent) const { d(indent, "Break"); }
void ContinueNode::dump(int indent) const { d(indent, "Continue"); }

void ForNode::dump(int indent) const
{
    d(indent, "For:%s", var.c_str());
    start->dump(indent + 1);
    end->dump(indent + 1);
    if (step)
        step->dump(indent + 1);
    body->dump(indent + 1);
}

void SwitchNode::dump(int indent) const
{
    d(indent, "Switch");
    value->dump(indent + 1);
    for (auto &c : cases)
    {
        d(indent + 1, "Case");
        c.first->dump(indent + 2);
        c.second->dump(indent + 2);
    }
    if (default_case)
    {
        d(indent + 1, "Default");
        default_case->dump(indent + 2);
    }
}

void LetNode::dump(int indent) const
{
    d(indent, "Let");
    for (auto &b : bindings)
    {
        pad(indent + 1);
        fputs(b.name.c_str(), stderr);
        if (b.type)
        {
            fputs(": ", stderr);
            fputs(b.type->to_str().c_str(), stderr);
        }
        if (b.init)
        {
            fputs(" = \n", stderr);
            b.init->dump(indent + 2);
        }
        else
            fputs(" (no init)\n", stderr);
    }
    body->dump(indent + 1);
}

void GlobalVarNode::dump(int indent) const
{
    d(indent, "GlobalVar");
    for (auto &b : bindings)
    {
        pad(indent + 1);
        fputs(b.name.c_str(), stderr);
        if (b.type)
        {
            fputs(": ", stderr);
            fputs(b.type->to_str().c_str(), stderr);
        }
        if (b.init)
        {
            fputs(" = \n", stderr);
            b.init->dump(indent + 2);
        }
        else
            fputs(" (no init)\n", stderr);
    }
}

void ArrNode::dump(int indent) const
{
    d(indent, "Array[%zu]", elems.size());
    for (auto &e : elems)
        e->dump(indent + 1);
}

void IdxNode::dump(int indent) const
{
    d(indent, "Index");
    target->dump(indent + 1);
    for (auto &idx : indices)
        idx->dump(indent + 1);
}

void FieldNode::dump(int indent) const
{
    d(indent, "Field:%s", field.c_str());
    target->dump(indent + 1);
}

void StructNode::dump(int indent) const
{
    d(indent, "Struct:%s", name.c_str());
    for (auto &f : fields)
    {
        pad(indent + 1);
        fprintf(stderr, "%s = \n", f.first.c_str());
        f.second->dump(indent + 2);
    }
}

void MatMulNode::dump(int indent) const
{
    d(indent, "MatMul");
    lhs->dump(indent + 1);
    rhs->dump(indent + 1);
}

void TransposeNode::dump(int indent) const
{
    d(indent, "Transpose");
    operand->dump(indent + 1);
}

void DotNode::dump(int indent) const
{
    d(indent, "Dot");
    lhs->dump(indent + 1);
    rhs->dump(indent + 1);
}

void GradNode::dump(int indent) const
{
    d(indent, "Grad:%s", fname.c_str());
    point->dump(indent + 1);
    if (eps)
        eps->dump(indent + 1);
}

void JacobianNode::dump(int indent) const
{
    d(indent, "Jacobian:%s", fname.c_str());
    point->dump(indent + 1);
    if (eps)
        eps->dump(indent + 1);
}
