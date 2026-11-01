import logging
from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

from torchcodec.encoders._audio_encoder import AudioEncoder
from yt_dlp import YoutubeDL

from yohane import Yohane
from yohane.audio import (
    ForcedAligner,
    HybridDemucsSeparator,
    Separator,
    TorchAudioForcedAligner,
    VocalRemoverSeparator,
    Wav2Vec2ForcedAligner,
)

DEFAULT_FORCED_ALIGNER = "NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn"

logger = logging.getLogger(__name__)


@contextmanager
def parse_song_argument(value: str) -> Generator[tuple[Path, Path]]:
    song_path = Path(value)

    if not song_path.is_file():
        logger.info("Song file not found, calling yt-dlp")
        song_path = ydl_download(value)

    output_path = song_path.with_suffix("")
    yield song_path, output_path


def ydl_download(value: str) -> Path:
    with YoutubeDL({"format_sort": ["res:1080", "vcodec:h264", "acodec:aac"]}) as ydl:
        info = ydl.extract_info(value)
        filename = ydl.prepare_filename(info)
        return Path(filename)


class SeparatorChoice(str, Enum):
    VocalRemover = "vocal-remover"
    HybridDemucs = "hybrid-demucs"


def get_separator(value: SeparatorChoice | None) -> Separator | None:
    match value:
        case SeparatorChoice.VocalRemover:
            return VocalRemoverSeparator()
        case SeparatorChoice.HybridDemucs:
            return HybridDemucsSeparator()
        case _:
            return None


def get_forced_aligner(value: str) -> ForcedAligner:
    if value.casefold() == "mms-fa":
        return TorchAudioForcedAligner()
    return Wav2Vec2ForcedAligner(value)


def save_separated_tracks(yohane: Yohane, output: Path):
    if yohane.vocals is not None:
        waveform, sample_rate = yohane.vocals
        filename = output.with_suffix(".vocals.wav")
        logger.info(f"Saving vocals track to {filename}")
        AudioEncoder(waveform, sample_rate=sample_rate).to_file(filename.as_posix())
    off_vocal = yohane.extract_off_vocal()
    if off_vocal is not None:
        waveform, sample_rate = off_vocal
        filename = output.with_suffix(".off_vocal.wav")
        logger.info(f"Saving off vocal track to {filename}")
        AudioEncoder(waveform, sample_rate=sample_rate).to_file(filename.as_posix())
