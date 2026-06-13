#include "diag.h"
#include <cstdarg>
#include <cstdio>

std::vector<ErrMsg> g_errors;

void err(const char *fmt, ...)
{
    ErrMsg e;
    e.pos = g_pos;
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    e.msg = buf;
    g_errors.push_back(e);
    fprintf(stderr, "Error at %d:%d: %s\n", e.pos.row, e.pos.col,
            e.msg.c_str());
}

bool has_err() { return !g_errors.empty(); }
