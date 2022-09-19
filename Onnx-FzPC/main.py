import argparse

from backend import prepare


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str, help="Path to the Model.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Prepare a BackendRep for the Model.
    backendrep = prepare(args.path)

    # Export the Model as Secfloat
    backendrep.export_model()


if __name__ == "__main__":
    main()
