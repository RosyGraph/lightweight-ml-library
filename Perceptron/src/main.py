import argparse

import numpy as np


def q2a():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for Perceptron experiments.")
    parser.add_argument(
        "--q2a",
        action="store_true",
        help="Run the standard Perceptron on the bank-note data set for T = 1 to 10. Stores the learned weight vector and average prediction error in the reports/ directory.",
    )
    args = parser.parse_args()
    if args.q2a:
        q2a()
