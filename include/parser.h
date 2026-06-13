#ifndef NABLA_PARSER_H
#define NABLA_PARSER_H

#include "ast.h"
#include "type.h"
#include <memory>
#include <string>
#include <vector>

extern int g_tok;
extern bool g_parsed_global_decl;
int advance();

std::unique_ptr<TypeNode> parse_type();
std::unique_ptr<Node> parse_expr(int rbp = 0);

std::unique_ptr<ProtoNode> parse_proto();
std::unique_ptr<FuncNode> parse_func();
std::unique_ptr<FuncNode> parse_toplevel();
std::unique_ptr<FuncNode> parse_toplevel_as(const std::string &name);
std::unique_ptr<ProtoNode> parse_foreign();
std::unique_ptr<StructDef> parse_struct_def();

#endif
