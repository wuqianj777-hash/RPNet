"""CCPD2020 license-plate detection and recognition package."""

from .ccpd import CCPD2020Dataset, decode_plate, parse_ccpd_filename
from .model import RPNet8

__all__ = ["CCPD2020Dataset", "RPNet8", "decode_plate", "parse_ccpd_filename"]

