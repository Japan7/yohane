import subprocess
from importlib.metadata import metadata
from pathlib import Path


def get_identifier():
    pkg_meta = metadata(__package__)
    version = pkg_meta["Version"]
    rev = get_rev_shorthash()
    if rev is not None:
        version += f"-{rev}"
    return f"{pkg_meta['Name']} {version} ({pkg_meta['Home-page']})"


def get_rev_shorthash():
    """Get the git revision hash of the current working directory."""
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    proc = subprocess.run(cmd, capture_output=True, cwd=Path(__file__).parent)
    if proc.returncode != 0:
        return None
    return proc.stdout.decode().strip()
