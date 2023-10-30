import logging
import os
from io import TextIOWrapper
from pathlib import Path

import click
import torchaudio

from yohane.audio import VOCALS_EXTRACTORS, compute_alignments
from yohane.lyrics import Lyrics
from yohane.subtitles import make_ass

logger = logging.getLogger(__name__)


@click.command(
    help="""
    SONG_FILE: Video or audio file of the song. If FFmpeg backend is not available (Windows), use a '.wav'.

    LYRICS_FILE: Text file which contains the lyrics.
    """,
)
@click.argument(
    "song_file",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "lyrics_file",
    type=click.File(),
)
@click.option(
    "-e",
    "--vocals-extractor",
    type=click.Choice((*VOCALS_EXTRACTORS, "None")),
    default=list(VOCALS_EXTRACTORS)[0],
    show_default=True,
    help="Vocals extractor to use. 'None' to disable.",
)
def main(song_file: Path, lyrics_file: TextIOWrapper, vocals_extractor: str):
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())

    waveform, sample_rate = torchaudio.load(song_file)  # type: ignore

    extractor_cls = VOCALS_EXTRACTORS.get(vocals_extractor, None)
    if extractor_cls is not None:
        extractor = extractor_cls()
        logger.info(f"Extracting vocals with {extractor=}")
        waveform, sample_rate = extractor(waveform, sample_rate)
        torchaudio.save(song_file.with_suffix(".vocals.wav"), waveform, sample_rate)  # type: ignore

    logger.info("Preparing lyrics")
    lyrics = Lyrics(lyrics_file.read())

    logger.info("Computing forced alignment")
    emission, token_spans = compute_alignments(waveform, sample_rate, lyrics.transcript)

    logger.info("Generating .ass")
    subs = make_ass(lyrics, waveform, sample_rate, emission, token_spans)
    subs_file = song_file.with_suffix(".ass")
    subs.save(subs_file.as_posix())
    logger.info(f"Saved to '{subs_file.as_posix()}'")


if __name__ == "__main__":
    main()
