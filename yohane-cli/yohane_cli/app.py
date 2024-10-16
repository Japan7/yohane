import logging
import os
from enum import Enum
from pathlib import Path
from typing import Annotated

import torchaudio
import typer

from yohane import Yohane
from yohane.audio import HybridDemucsSeparator, VocalRemoverSeparator

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

app = typer.Typer()


class SeparatorChoice(str, Enum):
    VocalRemover = "vocal-remover"
    HybridDemucs = "hybrid-demucs"
    Disable = "none"


@app.command(help="Generate a karaoke (full pipeline)")
def generate(
    song_file: Annotated[
        Path,
        typer.Argument(
            help="Video or audio file of the song. If FFmpeg backend is not available (Windows), provide a '.wav'.",
            exists=True,
            dir_okay=False,
        ),
    ],
    lyrics_file: Annotated[
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

    yohane.load_song(song_file)
    yohane.load_lyrics(lyrics_file.read())

    yohane.extract_vocals()
    save_separated_tracks(yohane, song_file)

    yohane.force_align()

    subs = yohane.make_subs()
    subs_file = song_file.with_suffix(".ass")
    subs.save(subs_file.as_posix())
    logger.info(f"Result saved to '{subs_file.as_posix()}'")


@app.command(help="Seperate vocals and instrumental tracks")
def separate(
    song_file: Annotated[
        Path,
        typer.Argument(
            help="Video or audio file of the song. If FFmpeg backend is not available (Windows), provide a '.wav'.",
            exists=True,
            dir_okay=False,
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
    yohane.load_song(song_file)

    yohane.extract_vocals()
    save_separated_tracks(yohane, song_file)


def get_separator(separator_choice: SeparatorChoice):
    match separator_choice:
        case SeparatorChoice.VocalRemover:
            return VocalRemoverSeparator()
        case SeparatorChoice.HybridDemucs:
            return HybridDemucsSeparator()
        case _:
            return None


def save_separated_tracks(yohane: Yohane, song_file: Path):
    if yohane.vocals is not None:
        waveform, sample_rate = yohane.vocals
        filename = song_file.with_suffix(".vocals.wav")
        logger.info(f"Saving vocals track to {filename}")
        torchaudio.save(filename, waveform.to("cpu"), sample_rate)
    if yohane.off_vocal is not None:
        waveform, sample_rate = yohane.off_vocal
        filename = song_file.with_suffix(".off_vocal.wav")
        logger.info(f"Saving off vocal track to {filename}")
        torchaudio.save(filename, waveform.to("cpu"), sample_rate)
