import logging
import os
from io import TextIOWrapper
from pathlib import Path

import click

from yohane.audio_processing import compute_alignments, prepare_audio
from yohane.subtitles import make_ass
from yohane.text_processing import Text

logger = logging.getLogger(__name__)


@click.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.argument("lyrics_file", type=click.File())
def main(lyrics_file: TextIOWrapper, audio_file: Path):
    lyrics_txt = Text.from_raw(lyrics_file.read())

    logger.info("Extracting vocals from audio...")
    waveform = prepare_audio(audio_file)

    logger.info("Aligning...")
    emission, token_spans = compute_alignments(waveform, lyrics_txt.transcript)

    logger.info("Making ASS...")
    subs = make_ass(lyrics_txt, waveform, emission, token_spans)
    subs_file = audio_file.with_suffix(".ass")
    subs.save(subs_file.as_posix())
    logger.info(f"Saved to {subs_file.as_posix()}")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    main()
