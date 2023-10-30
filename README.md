# yohane

<a target="_blank" href="https://colab.research.google.com/github/Japan7/yohane/blob/main/notebook/yohane.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Takes a song and its lyrics, extracts the vocals, splits the syllables and computes a forced alignment to generate a karaoke in an [Aegisub](https://aegisub.org) subtitles file (.ass).

## Getting Started

**Requirements:**

- [Python 3.10 or 3.11](https://www.python.org)
- [Poetry](https://python-poetry.org)
- [FFmpeg](https://ffmpeg.org)

```sh
git clone https://github.com/Japan7/yohane.git
cd yohane/
poetry install --only main
poetry run yohane
```

<!-- Tagged releases are also available on PyPI (`pip install yohane`). -->

For a ~4.5 min song, on a MacBook Pro 2018 (i5-8259U), CPU only:

- Vocals extraction takes ~5 min with vocal-remover
- Forced alignment takes 5-10 min

## Caveats

- This script is optimized for Japanese songs
- Torchaudio ffmpeg backend is not available on Windows: convert your song file to .wav before using yohane with `ffmpeg -i <src> <out>.wav`
- Long syllables at end of lines will often be truncated
- Forced alignment can't deal with overlapping vocals
- It is not fully accurate, you should still check and edit the result!

## Sample

[Aqours - PV - HAPPY PARTY TRAIN](https://hikari.butaishoujo.moe/v/9a11c0b1/Aqours%20-%20PV%20-%20HAPPY%20PARTY%20TRAIN.mp4) (hardsub extract, rev. [c43742c](https://github.com/Japan7/yohane/commit/c43742c1eb2ce9a86089a8d1b5fdc1fad458a91e))

## References

- [Forced alignment for multilingual data – Torchaudio documentation](https://pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html)
- [Music Source Separation with Hybrid Demucs – Torchaudio documentation](https://pytorch.org/audio/2.1.0/tutorials/hybrid_demucs_tutorial.html)
- [tsurumeso/vocal-remover](https://github.com/tsurumeso/vocal-remover)
- [Karaoke Mugen's auto-split.lua](https://docs.karaokes.moe/aegisub/auto-split.lua)
