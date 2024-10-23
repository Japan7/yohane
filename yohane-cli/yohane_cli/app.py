import logging
import os
from pathlib import Path
from typing import Annotated

import typer

from yohane import Yohane
from yohane_cli.audio import (
    SeparatorChoice,
    get_separator,
    parse_song_argument,
    save_separated_tracks,
)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command(help="Generate a karaoke (full pipeline)")
def generate(
    song: Annotated[
        Path,
        typer.Argument(
            parser=parse_song_argument,
            help="Video or audio file of the song. Can be an URL to download with yt-dlp.",
        ),
    ],
    lyrics: Annotated[
        typer.FileText,
        typer.Argument(
            help="Text file which contains the lyrics.",
        ),
    ],
    separator_choice: Annotated[
        SeparatorChoice,
        typer.Option(
            "--separator",
            "-s",
            help="Source separator to use. 'none' to disable.",
        ),
    ] = SeparatorChoice.VocalRemover,
):
    separator = get_separator(separator_choice)

    yohane = Yohane(separator)

    yohane.load_song(song)
    yohane.load_lyrics(lyrics.read())

    yohane.extract_vocals()
    save_separated_tracks(yohane, song)

    yohane.force_align()

    subs = yohane.make_subs()
    subs_file = song.with_suffix(".ass")
    subs.save(subs_file.as_posix())
    logger.info(f"Result saved to '{subs_file.as_posix()}'")


@app.command(help="Seperate vocals and instrumental tracks")
def separate(
    song: Annotated[
        Path,
        typer.Argument(
            parser=parse_song_argument,
            help="Video or audio file of the song. Can be an URL to download with yt-dlp.",
        ),
    ],
    separator_choice: Annotated[
        SeparatorChoice,
        typer.Option(
            "--separator",
            "-s",
            help="Source separator to use. 'none' to disable.",
        ),
    ] = SeparatorChoice.VocalRemover,
):
    separator = get_separator(separator_choice)
    if separator is None:
        raise RuntimeError("No separator selected")

    yohane = Yohane(separator)
    yohane.load_song(song)

    yohane.extract_vocals()
    save_separated_tracks(yohane, song)
