import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def parse_lyrics_argument(value: Path | None) -> str:
    if isinstance(value, Path):
        return value.read_text()

    logger.info("No lyrics text file, opening text editor")
    input = click.edit()
    if input is None:
        raise click.MissingParameter(param_type="argument", param_hint="'LYRICS_FILE'")
    return input
