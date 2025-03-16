import logging
import subprocess
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

import torchaudio
from yt_dlp import YoutubeDL

from yohane import Yohane
from yohane.audio import HybridDemucsSeparator, Separator, VocalRemoverSeparator

logger = logging.getLogger(__name__)


@contextmanager
def parse_song_argument(value: str) -> Generator[tuple[Path, Path]]:
    song_path = Path(value)

    if not song_path.is_file():
        logger.info("Song file not found, calling yt-dlp")
        song_path = ydl_download(value)

    output_path = song_path.with_suffix("")
    if "ffmpeg" in torchaudio.list_audio_backends() or song_path.suffix == ".wav":
        yield song_path, output_path
    else:
        logger.info(
            "Torchaudio FFmpeg backend is not available "
            "(macOS/Linux only, requires FFmpeg >=4.4,<7)"
        )
        logger.info("Converting the song to a .wav file with FFmpeg (via subprocess)")
        with ffmpeg_wav(song_path) as wav_path:
            yield wav_path, output_path


def ydl_download(value: str) -> Path:
    with YoutubeDL({"format_sort": ["res:1080", "vcodec:h264", "acodec:aac"]}) as ydl:
        info = ydl.extract_info(value)
        filename = ydl.prepare_filename(info)
        return Path(filename)


@contextmanager
def ffmpeg_wav(song_path: Path) -> Generator[Path]:
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp:
        proc = subprocess.run(["ffmpeg", "-y", "-i", song_path, temp.name])
        proc.check_returncode()
        yield Path(temp.name)


class SeparatorChoice(str, Enum):
    VocalRemover = "vocal-remover"
    HybridDemucs = "hybrid-demucs"
    Disable = "none"


def get_separator(separator_choice: SeparatorChoice) -> Separator | None:
    match separator_choice:
        case SeparatorChoice.VocalRemover:
            return VocalRemoverSeparator()
        case SeparatorChoice.HybridDemucs:
            return HybridDemucsSeparator()
        case _:
            return None


def save_separated_tracks(yohane: Yohane, output: Path):
    if yohane.vocals is not None:
        waveform, sample_rate = yohane.vocals
        filename = output.with_suffix(".vocals.wav")
        logger.info(f"Saving vocals track to {filename}")
        torchaudio.save(filename.as_posix(), waveform.to("cpu"), sample_rate)
    off_vocal = yohane.extract_off_vocal()
    if off_vocal is not None:
        waveform, sample_rate = off_vocal
        filename = output.with_suffix(".off_vocal.wav")
        logger.info(f"Saving off vocal track to {filename}")
        torchaudio.save(filename.as_posix(), waveform.to("cpu"), sample_rate)
