import argparse


def simple_parser():
    parser = argparse.ArgumentParser(
        description="LLM SDG PROJECT", epilog="Developped by Quinten"
    )

    ### LOGGING ###
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )
    parser.add_argument("--no-save",
        dest='save',
        action='store_false',
        help="No saving of files",
        
    )
    parser.add_argument("--save",
        default=True,
        help="No saving of files",
        
    )
    return parser
