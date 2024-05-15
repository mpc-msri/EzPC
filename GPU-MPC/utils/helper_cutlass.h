#include <cutlass/cutlass.h>

#define CUTLASS_CHECK(status)                                                                      \
{                                                                                                  \
    cutlass::Status error = status;                                                                \
    if (error != cutlass::Status::kSuccess) {                                                      \
        std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                  << std::endl;                                                                    \
    }                                                                                              \
}