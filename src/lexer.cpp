#include "lexer.h"
#include <cctype>
#include <cstdlib>

SourcePos g_pos;
static SourcePos s_lex = {1, 0};
std::string g_sym;
double g_float_val;
int64_t g_int_val;
std::string g_str_val;
bool g_skip_newlines = true;
std::string g_input_line;
size_t g_input_pos = 0;

void set_input_line(const std::string &line)
{
    g_input_line = line;
    g_input_pos = 0;
}

static int advance()
{
    int ch;
    if (g_input_pos < g_input_line.size())
    {
        ch = (unsigned char)g_input_line[g_input_pos++];
        if (ch == '\n' || ch == '\r')
        {
            s_lex.row++;
            s_lex.col = 0;
        }
        else
            s_lex.col++;
        return ch;
    }
    ch = getchar();
    if (ch == '\n' || ch == '\r')
    {
        s_lex.row++;
        s_lex.col = 0;
    }
    else
        s_lex.col++;
    return ch;
}

static int peek()
{
    if (g_input_pos < g_input_line.size())
        return (unsigned char)g_input_line[g_input_pos];
    int ch = getchar();
    if (ch != EOF)
        ungetc(ch, stdin);
    return ch;
}

static void skip_line()
{
    int ch;
    do
        ch = advance();
    while (ch != EOF && ch != '\n' && ch != '\r');
}

static void skip_block()
{
    int depth = 1;
    int prev = 0;
    while (depth > 0)
    {
        int ch = advance();
        if (ch == EOF)
            return;
        if (prev == '/' && ch == '*')
        {
            depth++;
            prev = 0;
        }
        else if (prev == '*' && ch == '/')
        {
            depth--;
            prev = 0;
        }
        else
            prev = ch;
    }
}

int next_token()
{
    static int s_last = ' ';

    while (g_skip_newlines ? isspace(s_last)
                           : (s_last == ' ' || s_last == '\t'))
        s_last = advance();

    g_pos = s_lex;

    if (s_last == '/' && peek() == '/')
    {
        skip_line();
        s_last = advance();
        return next_token();
    }

    if (s_last == '/' && peek() == '*')
    {
        advance();
        skip_block();
        s_last = advance();
        return next_token();
    }

    if ((unsigned char)s_last == 0xE2 && peek() == 0x88)
    {
        advance();
        if (peek() == 0x87)
        {
            advance();
            s_last = advance();
            return TOK_NABLA;
        }
        ungetc(0x88, stdin);
    }

    if (isalpha(s_last) || s_last == '_')
    {
        g_sym = s_last;
        while (isalnum((s_last = advance())) || s_last == '_')
            g_sym += s_last;

        if (g_sym == "fn")
            return TOK_DEF;
        if (g_sym == "extern")
            return TOK_EXT;
        if (g_sym == "if")
            return TOK_IF;
        if (g_sym == "then")
            return TOK_THEN;
        if (g_sym == "else")
            return TOK_ELSE;
        if (g_sym == "elif")
            return TOK_ELIF;
        if (g_sym == "for")
            return TOK_FOR;
        if (g_sym == "in")
            return TOK_IN;
        if (g_sym == "binary")
            return TOK_BIN;
        if (g_sym == "unary")
            return TOK_UNA;
        if (g_sym == "let")
            return TOK_VAR;
        if (g_sym == "mut")
            return TOK_MUT;
        if (g_sym == "while")
            return TOK_WHILE;
        if (g_sym == "break")
            return TOK_BREAK;
        if (g_sym == "continue")
            return TOK_CONTINUE;
        if (g_sym == "switch")
            return TOK_SWITCH;
        if (g_sym == "case")
            return TOK_CASE;
        if (g_sym == "default")
            return TOK_DEFAULT;
        if (g_sym == "struct")
            return TOK_STRUCT;
        if (g_sym == "record")
            return TOK_REC;
        if (g_sym == "true")
            return TOK_TRUE;
        if (g_sym == "false")
            return TOK_FALSE;
        if (g_sym == "int")
            return TOK_INT;
        if (g_sym == "dot")
            return TOK_DOT;
        if (g_sym == "grad")
            return TOK_GRAD;
        if (g_sym == "jacobian")
            return TOK_JACOBIAN;
        return TOK_SYM;
    }

    if (isdigit(s_last))
    {
        std::string buf;
        bool is_float = false;
        do
        {
            buf += s_last;
            s_last = advance();
            if (s_last == '.')
            {
                is_float = true;
                buf += s_last;
                s_last = advance();
            }
        } while (isdigit(s_last));

        if (is_float)
        {
            g_float_val = strtod(buf.c_str(), nullptr);
            return TOK_FLOAT;
        }
        g_int_val = strtoll(buf.c_str(), nullptr, 10);
        return TOK_INT;
    }

    if (s_last == '"')
    {
        g_str_val.clear();
        s_last = advance();
        while (s_last != '"' && s_last != EOF)
        {
            if (s_last == '\\')
            {
                s_last = advance();
                switch (s_last)
                {
                case 'n':
                    g_str_val += '\n';
                    break;
                case 't':
                    g_str_val += '\t';
                    break;
                case '\\':
                    g_str_val += '\\';
                    break;
                case '"':
                    g_str_val += '"';
                    break;
                default:
                    g_str_val += s_last;
                    break;
                }
            }
            else
                g_str_val += s_last;
            s_last = advance();
        }
        if (s_last == '"')
            s_last = advance();
        return TOK_STR;
    }

    if (s_last == '=' && peek() == '=')
    {
        advance();
        s_last = advance();
        return TOK_EQ;
    }
    if (s_last == '!' && peek() == '=')
    {
        advance();
        s_last = advance();
        return TOK_NE;
    }
    if (s_last == '<' && peek() == '=')
    {
        advance();
        s_last = advance();
        return TOK_LE;
    }
    if (s_last == '>' && peek() == '=')
    {
        advance();
        s_last = advance();
        return TOK_GE;
    }
    if (s_last == '&' && peek() == '&')
    {
        advance();
        s_last = advance();
        return TOK_AND;
    }
    if (s_last == '|' && peek() == '|')
    {
        advance();
        s_last = advance();
        return TOK_OR;
    }
    if (s_last == '-' && peek() == '>')
    {
        advance();
        s_last = advance();
        return TOK_ARROW;
    }
    if (s_last == '*' && peek() == '*')
    {
        advance();
        s_last = advance();
        return TOK_POW;
    }

    if (s_last == '#')
    {
        skip_line();
        s_last = advance();
        return next_token();
    }

    if (s_last == EOF)
        return TOK_EOF;

    int cur = s_last;
    if (cur == '\n' && !g_skip_newlines)
        s_last = ' ';
    else
        s_last = advance();
    return cur;
}
