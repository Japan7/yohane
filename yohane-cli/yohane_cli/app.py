import logging
import os
from enum import Enum
from pathlib import Path
from typing import Annotated

import torchaudio
import typer
from rich import print

from yohane import Yohane
from yohane.audio import (
    HybridDemucsVocalsExtractor,
    VocalRemoverVocalsExtractor,
)


class VocalsExtractorChoice(str, Enum):
    VocalRemover = "vocal-remover"
    HybridDemucs = "hybrid-demucs"
    none = "none"


VOCALS_EXTRACTORS = {
    VocalsExtractorChoice.VocalRemover: VocalRemoverVocalsExtractor,
    VocalsExtractorChoice.HybridDemucs: HybridDemucsVocalsExtractor,
}

app = typer.Typer()


@app.command()
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
    vocals_extractor: Annotated[
        VocalsExtractorChoice,
        typer.Option(
            "--vocals-extractor",
            "-x",
            help="Vocals extractor to use. 'none' to disable.",
        ),
    ] = VocalsExtractorChoice.VocalRemover,
):
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())

    extractor = None
    if vocals_extractor is not VocalsExtractorChoice.none:
        extractor_cls = VOCALS_EXTRACTORS[vocals_extractor]
        extractor = extractor_cls()

    yohane = Yohane(extractor)

    yohane.load_song(song_file)
    yohane.load_lyrics(lyrics_file.read())

    yohane.extract_vocals()
    if yohane.vocals is not None:
        waveform, sample_rate = yohane.vocals
        torchaudio.save(
            song_file.with_suffix(".vocals.wav"), waveform.to("cpu"), sample_rate
        )
    if yohane.off_vocal is not None:
        waveform, sample_rate = yohane.off_vocal
        torchaudio.save(
            song_file.with_suffix(".off_vocal.wav"), waveform.to("cpu"), sample_rate
        )

    yohane.force_align()

    subs = yohane.make_subs()

    subs_file = song_file.with_suffix(".ass")
    subs.save(subs_file.as_posix())
    print(f"Saved to '{subs_file.as_posix()}'")
