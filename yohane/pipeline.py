import logging
from pathlib import Path

import torch
from torchaudio.functional import TokenSpan, resample
from torchcodec.decoders import AudioDecoder

from yohane.audio import (
    ForcedAligner,
    Separator,
    TorchAudioForcedAligner,
    Wav2Vec2ForcedAligner,
)
from yohane.lyrics import Lyrics
from yohane.subtitles import make_ass

logger = logging.getLogger(__name__)


class Yohane:
    def __init__(
        self,
        *,
        separator: Separator | None,
        forced_aligner: str | None = None,
    ):
        self.separator = separator
        self.forced_aligner: ForcedAligner = (
            Wav2Vec2ForcedAligner(forced_aligner)
            if forced_aligner is not None
            else TorchAudioForcedAligner()
        )
        self.song: tuple[torch.Tensor, int] | None = None
        self.vocals: tuple[torch.Tensor, int] | None = None
        self.lyrics: Lyrics | None = None
        self.forced_alignment: tuple[torch.Tensor, list[list[TokenSpan]]] | None = None

    @property
    def forced_aligned_audio(self):
        return self.vocals if self.vocals is not None else self.song

    def load_song(self, song_file: Path):
        logger.info("Loading song")
        audio = AudioDecoder(song_file.as_posix())
        samples = audio.get_all_samples()
        waveform = samples.data
        if waveform.size(0) > 2:
            waveform = waveform.mean(dim=0, keepdim=True).repeat(2, 1)
        self.song = (waveform, samples.sample_rate)

    def extract_vocals(self):
        if self.separator is None:
            return
        logger.info(f"Extracting vocals with {self.separator=}")
        assert self.song
        self.vocals = self.separator(*self.song)

    def extract_off_vocal(self):
        if self.song is None or self.vocals is None:
            return
        song_waveform, song_sample_rate = self.song
        vocals_waveform, vocals_sample_rate = self.vocals
        vocals_waveform = resample(
            vocals_waveform, vocals_sample_rate, song_sample_rate
        )
        min_columns = min(song_waveform.size(1), vocals_waveform.size(1))
        song_waveform = song_waveform[:, :min_columns]
        vocals_waveform = vocals_waveform[:, :min_columns]
        return song_waveform - vocals_waveform, song_sample_rate

    def load_lyrics(self, lyrics_str: str):
        logger.info("Loading lyrics")
        self.lyrics = Lyrics(lyrics_str)

    def force_align(self):
        logger.info("Computing forced alignment")
        assert self.forced_aligned_audio is not None and self.lyrics is not None
        tokens = self.forced_aligner.tokenize(self.lyrics.transcript)
        self.forced_alignment = self.forced_aligner.align(
            tokens, *self.forced_aligned_audio
        )

    def make_subs(self):
        logger.info("Generating .ass")
        assert (
            self.lyrics is not None
            and self.forced_aligned_audio is not None
            and self.forced_alignment is not None
        )
        subs = make_ass(
            self.lyrics,
            *self.forced_aligned_audio,
            self.forced_aligner.tokenize,
            *self.forced_alignment,
        )
        return subs
