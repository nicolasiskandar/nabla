#include "type.h"

std::string TypeNode::to_str() const
{
    switch (tag)
    {
    case TypeTag::Int:
        return "int";
    case TypeTag::Float:
        return "float";
    case TypeTag::Bool:
        return "bool";
    case TypeTag::Str:
        return "string";
    case TypeTag::Void:
        return "void";
    case TypeTag::Array:
        return "[" + (elem ? elem->to_str() : "?") + "]";
    case TypeTag::Struct:
    {
        std::string s = "struct { ";
        for (auto &f : fields)
            s += f.first + ": " + (f.second ? f.second->to_str() : "?") + ", ";
        s += "}";
        return s;
    }
    case TypeTag::Matrix:
    {
        std::string s = "matrix";
        if (!matrix_shape.empty())
        {
            s += "[";
            for (size_t i = 0; i < matrix_shape.size(); ++i)
            {
                if (i > 0)
                    s += ", ";
                s += std::to_string(matrix_shape[i]);
            }
            s += "]";
        }
        return s;
    }
    case TypeTag::Named:
        return name;
    }
    return "unknown";
}
