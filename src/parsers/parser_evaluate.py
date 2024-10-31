import argparse


def fidelity_parser():
    parser = argparse.ArgumentParser(
        description="LLM SDG PROJECT", epilog="Developped by Quinten"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )

    parser.add_argument(
        "--synth-dataset", 
        required=True,
        help="Name of synthetic dataset to evaluate"
    )

    parser.add_argument(
        "--real-dataset",
        required=True,
        help="Name of real dataset to evaluate"
    )

    parser.add_argument(
        "--category-threshold",
        type=int,
        default=3,
        help="Maximum number of possible values for a discrete feature",
    )

    parser.add_argument(
        "--exp-name",
        help="Suffix added to the saved file during evaluation",
    )

    return parser


