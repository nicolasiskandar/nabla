#ifndef NABLA_DIAG_H
#define NABLA_DIAG_H

#include "lexer.h"
#include <string>
#include <vector>

struct ErrMsg
{
    SourcePos pos;
    std::string msg;
};

extern std::vector<ErrMsg> g_errors;

void err(const char *fmt, ...);
bool has_err();

#endif
