import logging
import subprocess
from enum import Enum
from pathlib import Path

import torchaudio
from yt_dlp import YoutubeDL

from yohane import Yohane
from yohane.audio import HybridDemucsSeparator, Separator, VocalRemoverSeparator

logger = logging.getLogger(__name__)


def parse_song_argument(value: str) -> Path:
    song_path = Path(value)

    if not song_path.is_file():
        logger.info(f"Using yt-dlp as {value} is not an existing file")
        song_path = ydl_download(value)

    if "ffmpeg" not in torchaudio.list_audio_backends():
        logger.info(
            "Torchaudio FFmpeg backend is not available "
            "(macOS/Linux only, requires FFmpeg >=4.4,<7)"
        )
        logger.info("Converting the song to a .wav file with FFmpeg (via subprocess)")
        song_path = ffmpeg_wav(song_path)

    return song_path


def ydl_download(value: str) -> Path:
    with YoutubeDL({"format_sort": ["res:1080", "vcodec:h264", "acodec:aac"]}) as ydl:
        info = ydl.extract_info(value)
        filename = ydl.prepare_filename(info)
        return Path(filename)


def ffmpeg_wav(song_path: Path) -> Path:
    wav_path = song_path.with_suffix(".wav")
    proc = subprocess.run(["ffmpeg", "-y", "-i", song_path, wav_path])
    proc.check_returncode()
    return wav_path


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


def save_separated_tracks(yohane: Yohane, song_path: Path):
    if yohane.vocals is not None:
        waveform, sample_rate = yohane.vocals
        filename = song_path.with_suffix(".vocals.wav")
        logger.info(f"Saving vocals track to {filename}")
        torchaudio.save(filename, waveform.to("cpu"), sample_rate)
    if yohane.off_vocal is not None:
        waveform, sample_rate = yohane.off_vocal
        filename = song_path.with_suffix(".off_vocal.wav")
        logger.info(f"Saving off vocal track to {filename}")
        torchaudio.save(filename, waveform.to("cpu"), sample_rate)
