import logging
from pathlib import Path

import torch
import torchaudio
from torchaudio.functional import TokenSpan

from yohane.audio import VocalsExtractor, compute_alignments
from yohane.lyrics import Lyrics
from yohane.subtitles import make_ass

logger = logging.getLogger(__name__)


class Yohane:
    def __init__(self, vocals_extractor: VocalsExtractor | None):
        self.vocals_extractor = vocals_extractor
        self.song: tuple[torch.Tensor, int] | None = None
        self.vocals: tuple[torch.Tensor, int] | None = None
        self.lyrics: Lyrics | None = None
        self.forced_alignment: tuple[torch.Tensor, list[list[TokenSpan]]] | None = None

    @property
    def forced_aligned_audio(self):
        return self.vocals if self.vocals is not None else self.song

    def load_song(self, song_file: Path):
        logger.info("Loading song")
        self.song = torchaudio.load(song_file)  # type: ignore

    def extract_vocals(self):
        if self.vocals_extractor is not None:
            logger.info(f"Extracting vocals with {self.vocals_extractor=}")
            assert self.song
            self.vocals = self.vocals_extractor(*self.song)

    def load_lyrics(self, lyrics_str: str):
        logger.info("Loading lyrics")
        self.lyrics = Lyrics(lyrics_str)

    def force_align(self):
        logger.info("Computing forced alignment")
        assert self.forced_aligned_audio is not None and self.lyrics is not None
        self.forced_alignment = compute_alignments(
            *self.forced_aligned_audio, self.lyrics.transcript
        )

    def make_subs(self):
        logger.info("Generating .ass")
        assert (
            self.lyrics is not None
            and self.forced_aligned_audio is not None
            and self.forced_alignment is not None
        )
        subs = make_ass(self.lyrics, *self.forced_aligned_audio, *self.forced_alignment)
        return subs
