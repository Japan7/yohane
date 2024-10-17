import re
from importlib.metadata import metadata


def get_identifier():
    if __package__ is None:
        raise ImportError("This module must be imported as a package")
    pkg_meta = metadata(__package__)
    identifier = f"{pkg_meta['Name']} {pkg_meta['Version']}"
    if urls := pkg_meta["Project-URL"]:
        if parsed := re.search(r"homepage, (\S+)\b", urls):
            identifier += f" ({parsed.group(1)})"
    return identifier
