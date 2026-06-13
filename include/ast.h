#ifndef NABLA_AST_H
#define NABLA_AST_H

#include "diag.h"
#include "type.h"
#include <memory>
#include <string>
#include <vector>

namespace llvm
{
class Value;
class Function;
} // namespace llvm

class Node
{
    SourcePos loc;

  public:
    Node(SourcePos loc = g_pos) : loc(loc) {}
    virtual ~Node() = default;
    virtual llvm::Value *compile() = 0;
    int line() const { return loc.row; }
    int col() const { return loc.col; }
    virtual void dump(int indent = 0) const;
};

class NumNode : public Node
{
    double val;

  public:
    NumNode(double val) : val(val) {}
    double value() const { return val; }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class IntNode : public Node
{
    int64_t val;

  public:
    IntNode(int64_t val) : val(val) {}
    int64_t value() const { return val; }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class StrNode : public Node
{
    std::string val;

  public:
    StrNode(std::string val) : val(std::move(val)) {}
    const std::string &value() const { return val; }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class BoolNode : public Node
{
    bool val;

  public:
    BoolNode(bool val) : val(val) {}
    bool value() const { return val; }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class SymNode : public Node
{
    std::string name;

  public:
    SymNode(SourcePos loc, std::string name) : Node(loc), name(std::move(name))
    {
    }
    const std::string &get_name() const { return name; }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class UnaryNode : public Node
{
    std::string op;
    std::unique_ptr<Node> operand;

  public:
    UnaryNode(std::string op, std::unique_ptr<Node> operand)
        : op(std::move(op)), operand(std::move(operand))
    {
    }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class BinaryNode : public Node
{
    std::string op;
    std::unique_ptr<Node> lhs, rhs;

  public:
    BinaryNode(SourcePos loc, std::string op, std::unique_ptr<Node> lhs,
               std::unique_ptr<Node> rhs)
        : Node(loc), op(std::move(op)), lhs(std::move(lhs)), rhs(std::move(rhs))
    {
    }
    const std::string &get_op() const { return op; }
    Node *get_lhs() const { return lhs.get(); }
    Node *get_rhs() const { return rhs.get(); }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class CallNode : public Node
{
    std::string callee;
    std::vector<std::unique_ptr<Node>> args;

  public:
    CallNode(SourcePos loc, std::string callee,
             std::vector<std::unique_ptr<Node>> args)
        : Node(loc), callee(std::move(callee)), args(std::move(args))
    {
    }
    const std::string &get_callee() const { return callee; }
    const std::vector<std::unique_ptr<Node>> &get_args() const { return args; }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class IfNode : public Node
{
    std::vector<std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>>> elifs;
    std::unique_ptr<Node> cond, then_branch, else_branch;

  public:
    IfNode(SourcePos loc, std::unique_ptr<Node> cond,
           std::unique_ptr<Node> then_branch,
           std::vector<std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>>>
               elifs,
           std::unique_ptr<Node> else_branch)
        : Node(loc), elifs(std::move(elifs)), cond(std::move(cond)),
          then_branch(std::move(then_branch)),
          else_branch(std::move(else_branch))
    {
    }
    Node *get_then_branch() const { return then_branch.get(); }
    Node *get_else_branch() const { return else_branch.get(); }
    const std::vector<std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>>> &
    get_elifs() const
    {
        return elifs;
    }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class WhileNode : public Node
{
    std::unique_ptr<Node> cond, body;

  public:
    WhileNode(std::unique_ptr<Node> cond, std::unique_ptr<Node> body)
        : cond(std::move(cond)), body(std::move(body))
    {
    }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class BreakNode : public Node
{
  public:
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class ContinueNode : public Node
{
  public:
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class ForNode : public Node
{
    std::string var;
    std::unique_ptr<Node> start, end, step, body;

  public:
    ForNode(std::string var, std::unique_ptr<Node> start,
            std::unique_ptr<Node> end, std::unique_ptr<Node> step,
            std::unique_ptr<Node> body)
        : var(std::move(var)), start(std::move(start)), end(std::move(end)),
          step(std::move(step)), body(std::move(body))
    {
    }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class SwitchNode : public Node
{
    std::unique_ptr<Node> value;
    std::vector<std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>>> cases;
    std::unique_ptr<Node> default_case;

  public:
    SwitchNode(
        std::unique_ptr<Node> value,
        std::vector<std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>>>
            cases,
        std::unique_ptr<Node> default_case)
        : value(std::move(value)), cases(std::move(cases)),
          default_case(std::move(default_case))
    {
    }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

struct LetBinding
{
    std::string name;
    std::unique_ptr<TypeNode> type;
    std::unique_ptr<Node> init;
};

class LetNode : public Node
{
    std::vector<LetBinding> bindings;
    std::unique_ptr<Node> body;

  public:
    LetNode(std::vector<LetBinding> bindings, std::unique_ptr<Node> body)
        : bindings(std::move(bindings)), body(std::move(body))
    {
    }
    const std::vector<LetBinding> &get_bindings() const { return bindings; }
    Node *get_body() const { return body.get(); }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class GlobalVarNode : public Node
{
    std::vector<LetBinding> bindings;
    bool mutable_;

  public:
    GlobalVarNode(std::vector<LetBinding> bindings, bool mutable_ = false)
        : bindings(std::move(bindings)), mutable_(mutable_)
    {
    }
    bool is_mutable() const { return mutable_; }
    const std::vector<LetBinding> &get_bindings() const { return bindings; }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class ArrNode : public Node
{
    std::vector<std::unique_ptr<Node>> elems;

  public:
    ArrNode(std::vector<std::unique_ptr<Node>> elems) : elems(std::move(elems))
    {
    }
    size_t size() const { return elems.size(); }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class IdxNode : public Node
{
    std::unique_ptr<Node> target;
    std::vector<std::unique_ptr<Node>> indices;

  public:
    IdxNode(std::unique_ptr<Node> target,
            std::vector<std::unique_ptr<Node>> indices)
        : target(std::move(target)), indices(std::move(indices))
    {
    }
    Node *get_target() const { return target.get(); }
    const std::vector<std::unique_ptr<Node>> &get_indices() const
    {
        return indices;
    }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class FieldNode : public Node
{
    std::unique_ptr<Node> target;
    std::string field;

  public:
    FieldNode(std::unique_ptr<Node> target, std::string field)
        : target(std::move(target)), field(std::move(field))
    {
    }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class StructNode : public Node
{
    std::string name;
    std::vector<std::pair<std::string, std::unique_ptr<Node>>> fields;

  public:
    StructNode(
        std::string name,
        std::vector<std::pair<std::string, std::unique_ptr<Node>>> fields)
        : name(std::move(name)), fields(std::move(fields))
    {
    }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

struct Param
{
    std::string name;
    std::unique_ptr<TypeNode> type;
};

class ProtoNode
{
    std::string name;
    std::vector<Param> params;
    std::unique_ptr<TypeNode> ret_type;
    bool is_operator;
    unsigned prec;
    int src_line;

  public:
    ProtoNode(SourcePos pos, std::string name, std::vector<Param> params,
              std::unique_ptr<TypeNode> ret_type, bool is_operator = false,
              unsigned prec = 0)
        : name(std::move(name)), params(std::move(params)),
          ret_type(std::move(ret_type)), is_operator(is_operator), prec(prec),
          src_line(pos.row)
    {
    }
    llvm::Function *compile();
    const std::string &get_name() const { return name; }
    const std::vector<Param> &get_params() const { return params; }
    const TypeNode *get_ret() const { return ret_type.get(); }
    bool is_unary() const { return is_operator && params.size() == 1; }
    bool is_binary() const { return is_operator && params.size() == 2; }
    char operator_char() const { return name.back(); }
    unsigned operator_prec() const { return prec; }
    int get_line() const { return src_line; }
};

class FuncNode
{
    std::unique_ptr<ProtoNode> proto;
    std::unique_ptr<Node> body;

  public:
    FuncNode(std::unique_ptr<ProtoNode> proto, std::unique_ptr<Node> body)
        : proto(std::move(proto)), body(std::move(body))
    {
    }
    Node *get_body() const { return body.get(); }
    llvm::Function *compile();
};

class MatMulNode : public Node
{
    std::unique_ptr<Node> lhs, rhs;
    int a_rows, a_cols, b_cols;

  public:
    MatMulNode(SourcePos loc, std::unique_ptr<Node> lhs,
               std::unique_ptr<Node> rhs, int a_rows = -1, int a_cols = -1,
               int b_cols = -1)
        : Node(loc), lhs(std::move(lhs)), rhs(std::move(rhs)), a_rows(a_rows),
          a_cols(a_cols), b_cols(b_cols)
    {
    }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;

    int get_rows() const { return a_rows; }
    int get_cols() const { return b_cols; }
    int get_inner() const { return a_cols; }
};

class TransposeNode : public Node
{
    std::unique_ptr<Node> operand;
    int in_rows, in_cols;

  public:
    TransposeNode(std::unique_ptr<Node> operand, int in_rows = -1,
                  int in_cols = -1)
        : operand(std::move(operand)), in_rows(in_rows), in_cols(in_cols)
    {
    }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;

    int get_rows() const { return in_cols; }
    int get_cols() const { return in_rows; }
};

class DotNode : public Node
{
    std::unique_ptr<Node> lhs, rhs;
    int vec_size;

  public:
    DotNode(SourcePos loc, std::unique_ptr<Node> lhs, std::unique_ptr<Node> rhs,
            int vec_size = -1)
        : Node(loc), lhs(std::move(lhs)), rhs(std::move(rhs)),
          vec_size(vec_size)
    {
    }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;
};

class GradNode : public Node
{
    std::string fname;
    std::unique_ptr<Node> point;
    std::unique_ptr<Node> eps;
    int pt_size;

  public:
    GradNode(SourcePos loc, std::string fname, std::unique_ptr<Node> point,
             std::unique_ptr<Node> eps, int pt_size = -1)
        : Node(loc), fname(std::move(fname)), point(std::move(point)),
          eps(std::move(eps)), pt_size(pt_size)
    {
    }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;

    int get_rows() const { return 1; }
    int get_cols() const { return pt_size; }
};

class JacobianNode : public Node
{
    std::string fname;
    std::unique_ptr<Node> point;
    std::unique_ptr<Node> eps;
    int out_size;
    int pt_size;

  public:
    JacobianNode(SourcePos loc, std::string fname, std::unique_ptr<Node> point,
                 std::unique_ptr<Node> eps, int pt_size = -1, int out_size = 1)
        : Node(loc), fname(std::move(fname)), point(std::move(point)),
          eps(std::move(eps)), out_size(out_size), pt_size(pt_size)
    {
    }
    llvm::Value *compile() override;
    void dump(int indent = 0) const override;

    int get_rows() const { return out_size; }
    int get_cols() const { return pt_size; }
};

struct StructDef
{
    std::string name;
    std::vector<std::pair<std::string, std::unique_ptr<TypeNode>>> fields;
};

#endif
