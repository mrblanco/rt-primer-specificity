"""
Shared utilities for the RT primer specificity analysis pipeline.
Adapted from nanobodypipeline helpers.
"""

import argparse
import gzip
import io
import typing


# magic number for the gzip file format
GZIP_MAGIC_NUMBER = b"\x1f\x8b"


class AutoCloseGzipFile(gzip.GzipFile):
    """
    Subclass of gzip.GzipFile whose close() method actually closes fileobj.
    """

    def close(self):
        if self.fileobj:
            self.fileobj.close()
        super().close()


def file_open(filename: str, mode: str = "rb", encoding: str = "utf-8") -> typing.IO:
    """
    Detect if a file is gzip-compressed and return an appropriate file object.
    Only supports "rb" and "rt" modes.
    """
    assert mode in ("rb", "rt"), 'file_open() only supports "rb" and "rt" modes'
    f = open(filename, "rb")
    first_two_bytes = f.peek(2)[:2]

    if len(first_two_bytes) != 2:
        first_two_bytes = f.read(2)
        f.seek(0)
    if first_two_bytes == GZIP_MAGIC_NUMBER:
        f = AutoCloseGzipFile(fileobj=f, mode="rb")
    if mode == "rt":
        return io.TextIOWrapper(f, encoding=encoding)
    else:
        return f


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(complement)[::-1]


def positive_int(value: str) -> int:
    """Check that a string represents a positive integer."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue
