import logging
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
from yohane_cli.lyrics import parse_lyrics_argument

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command(help="Generate a karaoke (full pipeline)")
def generate(
    song_file: Annotated[
        str,
        typer.Argument(
            help="Video or audio file of the song. Can be an URL to download with yt-dlp.",
        ),
    ],
    lyrics_file: Annotated[
        Path | None,
        typer.Argument(
            help="Text file which contains the lyrics. (Optional: otherwise, a text editor will open.)",
        ),
    ] = None,
    separator_choice: Annotated[
        SeparatorChoice,
        typer.Option(
            "--separator",
            "-s",
            help="Source separator to use. 'none' to disable.",
        ),
    ] = SeparatorChoice.VocalRemover,
):
    with parse_song_argument(song_file) as (song, output):
        lyrics = parse_lyrics_argument(lyrics_file)
        separator = get_separator(separator_choice)

        yohane = Yohane(separator)

        yohane.load_song(song)
        yohane.load_lyrics(lyrics)

        yohane.extract_vocals()
        save_separated_tracks(yohane, output)

        yohane.force_align()

        subs = yohane.make_subs()
        subs_file = output.with_suffix(".ass")
        subs.save(subs_file.as_posix())
        logger.info(f"Result saved to '{subs_file.as_posix()}'")


@app.command(help="Seperate vocals and instrumental tracks")
def separate(
    song_file: Annotated[
        str,
        typer.Argument(
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
    with parse_song_argument(song_file) as (song, output):
        separator = get_separator(separator_choice)
        if separator is None:
            raise RuntimeError("No separator selected")

        yohane = Yohane(separator)
        yohane.load_song(song)

        yohane.extract_vocals()
        save_separated_tracks(yohane, output)
