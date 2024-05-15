#pragma once

#include <sytorch/backend/cleartext.h>
#include <sytorch/backend/float.h>

template <typename T>
Backend<T>* defaultBackend()
{
    if constexpr (std::is_floating_point<T>::value) {
        return new FloatClearText<T>();
    } else {
        return new ClearText<T>();
    }
}
