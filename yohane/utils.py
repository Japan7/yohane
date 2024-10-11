from importlib.metadata import metadata


def get_identifier():
    if __package__ is None:
        raise ImportError("This module must be imported as a package")
    pkg_meta = metadata(__package__)
    return f"{pkg_meta['Name']} {pkg_meta['Version']} ({pkg_meta['Project-URL']})"
