import argparse, sys, os
from utils import logger

from backend import prepare


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str, help="Path to the Model.")
    parser.add_argument(
        "--generate", required=True, type=str, choices=["code", "executable"]
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Prepare a BackendRep for the Model.
    backendrep = prepare(args.path)

    # Export the Model as Secfloat and writes to a cpp file
    backendrep.export_model()

    if args.generate == "executable":
        logger.info("Starting Compilation.")
        os.system(f"lib_secfloat/compile_secfloat.sh {args.path[:-5]}_secfloat.cpp")
        logger.info(f"Output Binary generated : {args.path[:-5]}_secfloat.out")


if __name__ == "__main__":
    main()
