# yohane

Takes a song and its lyrics, extracts the vocals, splits the syllables and computes a forced alignment to generate a karaoke in an [Aegisub](https://aegisub.org) subtitles file (.ass).

## Getting Started

### Notebook

Open the [notebook](notebook/yohane.ipynb) in Google Colab to use their offered GPU resources:

<a target="_blank" href="https://colab.research.google.com/github/Japan7/yohane/blob/main/notebook/yohane.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The karaoke generation will take less than a minute in their environment.

### Local environment

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

## Caveats

- This script is optimized for Japanese songs
- Torchaudio ffmpeg backend is not available on Windows: convert your song file to .wav before using yohane with `ffmpeg -i <src> <out>.wav`
- Long syllables at end of lines will often be truncated
- Forced alignment can't deal with overlapping vocals
- It is not fully accurate, you should still check and edit the result!

## Recommended workflow

1. Get the song and its lyrics
2. Use the yohane notebook or the CLI locally to generate the karaoke file

In Aegisub:

3. Load the .ass and the video
4. Replace the _Default_ style with your own
5. Due to the normalization during the process, lines are lowercased and special characters have been removed: use the original lines in comments to fix the timed lines
6. Subtitle > Select Lines… > check _Comments_ and _Set selection_ > OK and delete the selected lines
7. Listen to each line and fix their End time
8. Watch each line with the video and fix/merge syllable timings in karaoke mode if necessary

You're quite done here. Maybe try further styling?

## Sample

[Aqours - PV - HAPPY PARTY TRAIN](https://hikari.butaishoujo.moe/v/9a11c0b1/Aqours%20-%20PV%20-%20HAPPY%20PARTY%20TRAIN.mp4) (unedited hardsub extract, rev. [c43742c](https://github.com/Japan7/yohane/commit/c43742c1eb2ce9a86089a8d1b5fdc1fad458a91e))

## References

- [Forced alignment for multilingual data – Torchaudio documentation](https://pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html)
- [Music Source Separation with Hybrid Demucs – Torchaudio documentation](https://pytorch.org/audio/2.1.0/tutorials/hybrid_demucs_tutorial.html)
- [tsurumeso/vocal-remover](https://github.com/tsurumeso/vocal-remover)
- [Karaoke Mugen's auto-split.lua](https://docs.karaokes.moe/aegisub/auto-split.lua)
