#ifndef NABLA_LEXER_H
#define NABLA_LEXER_H

#include "token.h"
#include <string>

struct SourcePos
{
    int row;
    int col;
};

extern SourcePos g_pos;
extern std::string g_sym;
extern double g_float_val;
extern int64_t g_int_val;
extern std::string g_str_val;
extern bool g_skip_newlines;
extern std::string g_input_line;
extern size_t g_input_pos;

void set_input_line(const std::string &line);
int next_token();

#endif
