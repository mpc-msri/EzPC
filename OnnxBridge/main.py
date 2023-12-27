"""
Authors: Saksham Gupta.
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse, sys, os
from sys import argv
from utils import logger

from backend import prepare


def parse_args():
    backend = [
        "CLEARTEXT_LLAMA",
        "LLAMA",
        "SECFLOAT",
        "SECFLOAT_CLEARTEXT",
        "CLEARTEXT_fp",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str, help="Path to the Model.")
    parser.add_argument(
        "--scale",
        required=any(b in argv for b in [backend[0], backend[1]]),
        type=int,
        help="Scale for computation",
        default=0,
    )
    parser.add_argument(
        "--bitlength",
        required=any(b in argv for b in [backend[1]]),
        type=int,
        help="Bitlength for computation(required for LLAMA only)",
        default=0,
    )
    parser.add_argument(
        "--backend",
        required=True,
        type=str,
        choices=backend,
        help="Backend to compile model to.",
    )
    parser.add_argument(
        "--generate",
        required=True,
        type=str,
        choices=["code", "executable"],
        default="code",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Prepare a IR for the Model.
    backendrep = prepare(args.path, args.backend)
    mode = "u64" if args.backend == "LLAMA" else "i64"

    # Export the Model as Secfloat and writes to a cpp file
    if args.backend in ["CLEARTEXT_LLAMA", "LLAMA", "CLEARTEXT_fp"]:
        main_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(main_path, "LLAMA")
        backendrep.export_model(mode, args.scale, args.bitlength, args.backend)
    elif args.backend in ["SECFLOAT", "SECFLOAT_CLEARTEXT"]:
        main_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(main_path, "Secfloat")
        backendrep.export_model(file_path)

    if args.generate == "executable":
        logger.info("Starting Compilation.")
        if args.backend in ["SECFLOAT", "SECFLOAT_CLEARTEXT"]:
            ct = "" if args.backend == "SECFLOAT" else "_ct"
            os.system(
                f"{file_path}/compile_secfloat.sh {args.path[:-5]}_secfloat{ct}.cpp"
            )
        elif args.backend in ["CLEARTEXT_LLAMA", "LLAMA", "CLEARTEXT_fp"]:
            os.system(
                f"{file_path}/compile_llama.sh {args.path[:-5]}_{args.backend}_{args.scale}.cpp"
            )
        logger.info(
            f"Output Binary generated : {args.path[:-5]}_{args.backend}_{args.scale}.cpp"
        )


if __name__ == "__main__":
    main()
