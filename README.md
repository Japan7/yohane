# yohane

Takes a song and its lyrics, extracts the vocals, splits the syllables and computes a forced alignment to generate a karaoke in an [Aegisub](https://aegisub.org) subtitles file (.ass).

## Getting Started

### Notebook

Open the [notebook](notebook/yohane.ipynb) in Google Colab to use their offered GPU resources:

<a target="_blank" href="https://colab.research.google.com/github/Japan7/yohane/blob/main/notebook/yohane.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The full pipeline will be completed in less than a minute in their environment.

### Local environment

#### With `uv`

**Requirements:**

- [`uv`](https://github.com/astral-sh/uv)
- [FFmpeg](https://ffmpeg.org)

```sh
uvx --from git+https://github.com/Japan7/yohane.git[cli] --python 3.12 yohane --help
```

#### With `pixi`

**Requirement:** [`pixi`](https://prefix.dev)

```sh
git clone https://github.com/Japan7/yohane.git
cd yohane/
pixi run yohane --help
```

## Caveats

- Yohane's syllable splitting is only optimized for Japanese lyrics at the moment
- Syllables at the end of lines are often shortened
- Forced alignment can't deal with overlapping vocals
- It is not fully accurate, you should still check and edit the result!

## Recommended workflow

1. Get the song and its lyrics
2. Use the yohane notebook or the CLI locally to generate the karaoke file

In Aegisub:

1. Load the .ass and the video
2. Replace the _Default_ style with your own
3. Due to the normalization during the process, lines are lowercased and special characters have been removed: use the original lines in comments to fix the timed lines
4. Subtitle > Select Lines… > check _Comments_ and _Set selection_ > OK and delete the selected lines
5. Listen to each line and fix their End time
6. Add a 1s karaoke lead-in to every line
7. Iterate over each line in karaoke mode and merge/fix syllable timings

<img src="https://github.com/user-attachments/assets/614cd8ca-d471-447c-8596-4ac800d690cf" width="25%" >

## Sample

- **CRYCHIC - Haruhikage (THE FIRST TAKE)** ([2025.03.0](https://github.com/Japan7/yohane/releases/tag/2025.03.0)):
  [Video](https://www.youtube.com/watch?v=NJ1tne9u8YM),
  [Output](<samples/CRYCHIC - 春日影 ⧸ THE FIRST TAKE [NJ1tne9u8YM].ass>)
- **KAF x ZOOKARADERU - Himitsu no Kotoba** ([2024.10.5](https://github.com/Japan7/yohane/releases/tag/2024.10.5)):
  [Video](https://youtu.be/rnpL3ZugPLc?si=sXZH_EPLt3jaQq9K),
  [Output](<samples/【組曲】花譜×ズーカラデル # 124「秘密の言葉」【オリジナルMV】 [rnpL3ZugPLc].ass>)
- **Aqours - HAPPY PARTY TRAIN** ([c43742c](https://github.com/Japan7/yohane/commit/c43742c1eb2ce9a86089a8d1b5fdc1fad458a91e)):
  [Hardsubbed Video](https://hikari.butaishoujo.moe/v/9a11c0b1/Aqours%20-%20PV%20-%20HAPPY%20PARTY%20TRAIN.mp4)

## References

- [Forced alignment for multilingual data – Torchaudio documentation](https://pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html)
- [Music Source Separation with Hybrid Demucs – Torchaudio documentation](https://pytorch.org/audio/2.1.0/tutorials/hybrid_demucs_tutorial.html)
- [tsurumeso/vocal-remover](https://github.com/tsurumeso/vocal-remover)
- [Karaoke Mugen's auto-split.lua](https://docs.karaokes.moe/aegisub/auto-split.lua)
