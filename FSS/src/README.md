## Files

* `deps/` - This directory contains the external code from [ladnir/cryptoTools](https://github.com/ladnir/cryptoTools) on which our codebase depends.
* `add.cpp` and `add.h` - This contains the implementation of addition of two masked integers. It is used to replace the `+` operator in EzPC.
* `api.cpp` and `api.h` - This file contains the implementation of the Athos API using FSS. Every extern function called by Athos is implemented here. All the functions work in fixed bitwidth.
* `api_varied.cpp` and `api_varied.h` - This file contains the implementation of the SeeDot mixed-bitwidth API using FSS. This means that all the extern functions that SeeDot dumps are implemented in this file. Some API endpoints also contain an implicit implementation of many FSS Gates in the paper.
* `ArgMapping.h` - Argument parser for LLAMA binary.
* `array.h` - Contains convenient functions and macros for array allocation and indexing
* `comms.cpp` and`comms.h` - Contains helper functions for transfering data between dealer-evaluator (offline communication) and evaluator-evaluator (online communication). Contains wrappers for transferring different kinds of keys using fundamental transfer of `block`s and `GroupElement`s.
* `config.h` - Contains the compile time configuration macros.
* `conv.cpp` and `conv.h` - Contains the implementation of FSS gates for Convolution and Matmul.
* `dcf.cpp` and `dcf.h` - Contains the implementation of FSS scheme for DCF from [BCG+21](https://eprint.iacr.org/2020/1392).
* `fss.h` - Public header file for FSS.
* `GroupElement.h` - Contains the definition of the GroupElement class - a wrapper over `uint64_t`. 
* `input_prng.cpp` and `input_prng.h` - Contains the implementation of the input layers which uses a PRNG to compress the keysize required for input layers.
* `keypack.h` - Contains structures of different FSS Keys.
* `lib.cpp`, `lib.h` and `lib.ezpc` - Contains implementations of functions which can be written as wrappers over API endpoints. `lib.cpp` and `lib.h` are generated from the `lib.ezpc` file by running `fssc --bitlen 64 --l lib.ezpc`.
* `mult.cpp` and `mult.h` - Contains the implementation of multiplication of two masked integers. It is used to replace the `*` operator in EzPC. It contains both uniform bitwidth (from BCG+21) and varying bitwidth (from LLAMA paper).
* `pubdiv.cpp` and `pubdiv.h` - Contains the implementation of FSS gates for Public Division and ARS along with the FSS Gates for Signed Comparison and Interval Containment as the FSS Gate for Public Divison depends on these two gates.
* `spline.cpp` and `spline.h` - Contains implementation of FSS gates for Spline Evaluation. This gate is used to implement FSS Gates for ReLU and math functions like Sigmoid, Tanh and Reciprocal-Squareroot.
* `utils.cpp` and `utils.h` - Contains helper functions.

## Protocols

We have implemented the following protocols from the [paper](https://eprint.iacr.org/2022/793.pdf) in the respective files:

1. Sign Extension (Figure 1): `internalExtend` function in `api_varied.cpp`
2. Truncate-Reduce (Figure 2): `internalTruncateAndFix` function in `api_varied.cpp`
3. Signed Multiplication (Figure 3): `new_mult_signed_gen` and `new_mult_signed_eval` from `mult.cpp`
4. Unsigned Multiplication (Figure 4): `new_mult_unsigned_gen` and `new_mult_unsigned_eval` from `mult.cpp`
5. Signed Division (Figure 5): `keyGenSignedPublicDiv`, `evalSignedPublicDiv_First` and `evalSignedPublicDiv_Second` from `pubdiv.cpp`
6. Mixed Bitwidth Splines (Figure 6): `keygenSigmoid` and `evalSigmoid` (name is a bit misleading but same function is used for all math functions with different coefficients) from `spline.cpp`
