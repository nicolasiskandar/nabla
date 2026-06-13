#ifndef NABLA_TYPE_H
#define NABLA_TYPE_H

#include <memory>
#include <string>
#include <vector>

enum class TypeTag
{
    Int,
    Float,
    Bool,
    Str,
    Void,
    Array,
    Struct,
    Named,
    Matrix,
};

struct TypeNode
{
    TypeTag tag;
    std::unique_ptr<TypeNode> elem; // for array
    std::vector<std::pair<std::string, std::unique_ptr<TypeNode>>>
        fields;                             // for struct
    std::string name;                       // for named
    int matrix_rows = -1, matrix_cols = -1; // for matrix
    std::vector<int> matrix_shape;          // for n-dimensional matrix

    TypeNode(TypeTag tag) : tag(tag) {}

    static std::unique_ptr<TypeNode> make_int()
    {
        return std::make_unique<TypeNode>(TypeTag::Int);
    }
    static std::unique_ptr<TypeNode> make_float()
    {
        return std::make_unique<TypeNode>(TypeTag::Float);
    }
    static std::unique_ptr<TypeNode> make_bool()
    {
        return std::make_unique<TypeNode>(TypeTag::Bool);
    }
    static std::unique_ptr<TypeNode> make_str()
    {
        return std::make_unique<TypeNode>(TypeTag::Str);
    }
    static std::unique_ptr<TypeNode> make_void()
    {
        return std::make_unique<TypeNode>(TypeTag::Void);
    }
    static std::unique_ptr<TypeNode> make_array(std::unique_ptr<TypeNode> elem)
    {
        auto t = std::make_unique<TypeNode>(TypeTag::Array);
        t->elem = std::move(elem);
        return t;
    }
    static std::unique_ptr<TypeNode> make_named(const std::string &name)
    {
        auto t = std::make_unique<TypeNode>(TypeTag::Named);
        t->name = name;
        return t;
    }
    static std::unique_ptr<TypeNode>
    make_matrix(const std::vector<int> &shape = {})
    {
        auto t = std::make_unique<TypeNode>(TypeTag::Matrix);
        t->matrix_shape = shape;
        if (shape.size() >= 2)
        {
            t->matrix_rows = shape[0];
            t->matrix_cols = shape[1];
        }
        return t;
    }

    std::string to_str() const;
};

#endif
