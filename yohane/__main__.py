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
    compute_alignments,
)
from yohane.lyrics import Lyrics
from yohane.subtitles import make_ass

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

    lyrics_str = lyrics_file.read()

    extractor_cls = CLI_VOCALS_EXTRACTORS_OPTS[vocals_extractor]
    extractor = extractor_cls() if extractor_cls is not None else None

    subs = main(song_file, lyrics_str, extractor)

    subs_file = song_file.with_suffix(".ass")
    subs.save(subs_file.as_posix())
    logger.info(f"Saved to '{subs_file.as_posix()}'")


def main(song_file: Path, lyrics_str: str, vocals_extractor: VocalsExtractor | None):
    waveform, sample_rate = torchaudio.load(song_file)  # type: ignore

    if vocals_extractor is not None:
        logger.info(f"Extracting vocals with {vocals_extractor=}")
        waveform, sample_rate = vocals_extractor(waveform, sample_rate)
        torchaudio.save(song_file.with_suffix(".vocals.wav"), waveform, sample_rate)  # type: ignore

    logger.info("Preparing lyrics")
    lyrics = Lyrics(lyrics_str)

    logger.info("Computing forced alignment")
    emission, token_spans = compute_alignments(waveform, sample_rate, lyrics.transcript)

    logger.info("Generating .ass")
    subs = make_ass(lyrics, waveform, sample_rate, emission, token_spans)
    return subs


if __name__ == "__main__":
    cli()
