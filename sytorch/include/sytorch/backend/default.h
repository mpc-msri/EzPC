#pragma once

#include <sytorch/backend/cleartext.h>
#include <sytorch/backend/float.h>

template <typename T>
Backend<T> *defaultBackend()
{
    if constexpr (std::is_floating_point<T>::value)
    {
        return new FloatClearText<T>();
    }
    else
    {
        return new ClearText<T>();
    }
}

template <typename T>
inline T type_cast(float val);

template <>
float type_cast(float val)
{
    return val;
}

template <>
i64 type_cast(float val)
{
    return (i64)val;
}

template <>
u64 type_cast(float val)
{
    return (u64(i64(val)));
}