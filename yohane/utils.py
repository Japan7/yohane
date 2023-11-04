from importlib.metadata import metadata


def get_identifier():
    pkg_meta = metadata(__package__)
    return f"{pkg_meta['Name']} {pkg_meta['Version']} ({pkg_meta['Home-page']})"
