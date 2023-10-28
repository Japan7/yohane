# yohane

Takes a song and its lyrics as input, extracts the vocals, splits the syllabes and computes a forced alignment to generate a karaoke in an [Aegisub](https://aegisub.org) subtitles file (.ass).

This script is optimized for Japanese songs.

## Getting Started

**Requirements:**

- [Python 3.11](https://www.python.org)
- [Poetry](https://python-poetry.org)
- [FFmpeg](https://ffmpeg.org)

```sh
poetry install
poetry run python3 -m yohane
```

For a ~4 min song, on a MacBook Pro 2018 (i5-8259U), CPU only:

- Vocals extraction takes ~5 min
- Forced alignment takes ~10 min

## Sample

[Aqours - PV - HAPPY PARTY TRAIN](https://hikari.butaishoujo.moe/v/d6730f16/Aqours%20-%20PV%20-%20HAPPY%20PARTY%20TRAIN%20%28extract%29.mp4) (hardsub extract, rev. [6cfa494](https://github.com/Japan7/yohane/commit/6cfa4944069b576b9c8accc3e90c684db02f9947))

## References

- [Forced alignment for multilingual data – Torchaudio documentation](https://pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html)
- [tsurumeso/vocal-remover](https://github.com/tsurumeso/vocal-remover)
- [Karaoke Mugen's auto-split.lua](https://docs.karaokes.moe/aegisub/auto-split.lua)
