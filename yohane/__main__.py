import logging
import os
from io import TextIOWrapper
from pathlib import Path

import click
import torchaudio

from yohane.audio import (
    HybridDemucsVocalsExtractor,
    VocalRemoverVocalsExtractor,
    VocalsExtractor,
)
from yohane.pipeline import Yohane

logger = logging.getLogger(__name__)


CLI_VOCALS_EXTRACTORS_OPTS: dict[str, type[VocalsExtractor] | None] = {
    "VocalRemover": VocalRemoverVocalsExtractor,
    "HybridDemucs": HybridDemucsVocalsExtractor,
    "None": None,
}


@click.command(
    help="""
    SONG_FILE: Video or audio file of the song. If FFmpeg backend is not available (Windows), use a '.wav'.

    LYRICS_FILE: Text file which contains the lyrics.
    """,
)
@click.argument(
    "song_file",
    type=click.File(),
)
@click.argument(
    "lyrics_file",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-e",
    "--vocals-extractor",
    type=click.Choice(list(CLI_VOCALS_EXTRACTORS_OPTS.keys())),
    default=list(CLI_VOCALS_EXTRACTORS_OPTS)[0],
    show_default=True,
    help="Vocals extractor to use. 'None' to disable.",
)
def cli(song_file: Path, lyrics_file: TextIOWrapper, vocals_extractor: str):
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())

    extractor_cls = CLI_VOCALS_EXTRACTORS_OPTS[vocals_extractor]
    extractor = extractor_cls() if extractor_cls is not None else None

    yohane = Yohane(extractor)

    yohane.load_song(song_file)
    yohane.load_lyrics(lyrics_file.read())

    yohane.extract_vocals()
    if yohane.vocals is not None:
        waveform, sample_rate = yohane.vocals
        waveform = waveform.to("cpu")
        torchaudio.save(song_file.with_suffix(".vocals.wav"), waveform, sample_rate)  # type: ignore

    yohane.force_align()

    subs = yohane.make_subs()

    subs_file = song_file.with_suffix(".ass")
    subs.save(subs_file.as_posix())
    logger.info(f"Saved to '{subs_file.as_posix()}'")


if __name__ == "__main__":
    cli()
