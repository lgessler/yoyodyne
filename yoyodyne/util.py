"""Utilities."""

import argparse
import sys
import pickle


def log_info(msg: str) -> None:
    """Logs msg to sys.stderr.

    We can additionally consider logging to a file, or getting a handle to the
    PL logger.

    Args:
        msg (str): the message to log.
    """
    print(msg, file=sys.stderr)


def log_arguments(args: argparse.Namespace) -> None:
    """Logs non-null arguments via log_info.

    Args:
        args (argparse.Namespace).
    """
    log_info("Arguments:")
    for arg, val in vars(args).items():
        if val is None:
            continue
        log_info(f"\t{arg}: {val!r}")


def pickle_load(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)

def pickle_dump(fp, o):
    with open(fp, 'wb') as f:
        pickle.dump(o, f)