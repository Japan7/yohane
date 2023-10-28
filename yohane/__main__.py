from io import TextIOWrapper
from pathlib import Path

import click

from yohane.audio_processing import compute_alignments, load_audio
from yohane.subtitles import make_ass
from yohane.text_processing import Text


@click.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.argument("lyrics_file", type=click.File())
def main(lyrics_file: TextIOWrapper, audio_file: Path):
    lyrics_txt = Text.from_raw(lyrics_file.read())
    waveform = load_audio(audio_file)

    print("Aligning...")
    emission, token_spans = compute_alignments(waveform, lyrics_txt.transcript)

    print("Making ASS...")
    subs = make_ass(lyrics_txt, waveform, emission, token_spans)
    subs_file = audio_file.with_suffix(".ass")
    subs.save(subs_file.as_posix())
    print(f"Saved to {subs_file.as_posix()}")


if __name__ == "__main__":
    main()
