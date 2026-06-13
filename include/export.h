#ifndef NABLA_EXPORT_H
#define NABLA_EXPORT_H

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#endif
