#ifndef NABLA_TOKEN_H
#define NABLA_TOKEN_H

#include <string>

enum Token
{
    TOK_EOF = -1,

    TOK_DEF = -2,
    TOK_EXT = -3,
    TOK_SYM = -4,
    TOK_FLOAT = -5,

    TOK_IF = -6,
    TOK_THEN = -7,
    TOK_ELSE = -8,
    TOK_ELIF = -9,
    TOK_FOR = -10,
    TOK_IN = -11,
    TOK_BIN = -12,
    TOK_UNA = -13,
    TOK_VAR = -14,

    TOK_WHILE = -15,
    TOK_BREAK = -16,
    TOK_CONTINUE = -17,

    TOK_SWITCH = -18,
    TOK_CASE = -19,
    TOK_DEFAULT = -20,

    TOK_STRUCT = -21,

    TOK_STR = -22,
    TOK_INT = -23,
    TOK_TRUE = -24,
    TOK_FALSE = -25,

    TOK_EQ = -26,
    TOK_NE = -27,
    TOK_LE = -28,
    TOK_GE = -29,
    TOK_AND = -30,
    TOK_OR = -31,
    TOK_ARROW = -32,

    TOK_REC = -33,

    TOK_DOT = -34,
    TOK_GRAD = -35,
    TOK_JACOBIAN = -36,
    TOK_NABLA = -37,
    TOK_MUT = -38,
    TOK_POW = -39,
};

std::string token_desc(int token);

#endif
