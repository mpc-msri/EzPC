import argparse, sys, os
from utils import logger

from backend import prepare


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str, help="Path to the Model.")
    parser.add_argument(
        "--generate", required=True, type=str, choices=["code", "executable"]
    )
    parser.add_argument(
        "--backend", required=True, type=str, choices=["SECFLOAT", "SECFLOAT_CLEARTEXT"]
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Prepare a BackendRep for the Model.
    main_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(main_path, "Secfloat")
    backendrep = prepare(args.path, args.backend)

    # Export the Model as Secfloat and writes to a cpp file
    backendrep.export_model(file_path)

    ct = "" if args.backend == "SECFLOAT" else "_ct"

    if args.generate == "executable":
        logger.info("Starting Compilation.")
        os.system(f"{file_path}/compile_secfloat.sh {args.path[:-5]}_secfloat{ct}.cpp")
        logger.info(f"Output Binary generated : {args.path[:-5]}_secfloat{ct}.out")


if __name__ == "__main__":
    main()
