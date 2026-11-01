# yohane

Yohane takes a song and its lyrics, optionally extracts the vocals, splits the lyrics into syllables, and computes a forced alignment to generate karaoke subtitles in an [Aegisub](https://aegisub.org) subtitle file (.ass).

By default, it now uses our [NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn](https://huggingface.co/NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn) fine-tuned model to produce high-quality alignments for Japanese romaji lyrics.

## Getting Started

### Google Colab (Free GPU)

Open the provided [notebook](notebook/yohane.ipynb) in Google Colab and use their free T4 GPUs:

<a target="_blank" href="https://colab.research.google.com/github/Japan7/yohane/blob/main/notebook/yohane.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Hugging Face Space (CPU-only)

<a target="_blank" href="https://nextfire-yohane.hf.space">
  <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg" alt="Open In HF Spaces"/>
</a>

### Local Environment

#### With `uv` (PyPI)

**Requirements:**

- [`uv`](https://github.com/astral-sh/uv)
- [FFmpeg](https://ffmpeg.org)

```sh
uvx --from git+https://github.com/Japan7/yohane.git[cli] --python 3.14 yohane --help
```

#### With `pixi` (Conda)

**Requirement:** [`pixi`](https://prefix.dev)

```sh
git clone https://github.com/Japan7/yohane.git
cd yohane/
pixi run -e [default|cpu|cu126|cu130] yohane --help
```

## Caveats

- Yohane's syllable splitting is currently optimized for Japanese lyrics only.
- Syllables at the ends of lines are often shortened (this occurs less frequently with our [fine-tuned model](https://huggingface.co/NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn)).
- Forced alignment cannot handle overlapping vocals.
- The results are not always fully accurate, so you should still review and edit them!

## Recommended workflow

1. Get the song and its lyrics.
2. Use the Yohane notebook or CLI locally to generate the karaoke file.

In Aegisub

1. Load the `.ass` file and the video.
2. Replace the _Default_ style with your own.
3. Due to normalization during processing, lines are lowercased and special characters are removed. Use the original lyrics stored in comment lines to restore the timed lines.
4. Go to **Subtitle > Select Lines…**, check **Comments** and **Set selection**, then click **OK** and delete the selected lines.
5. Listen to each line and adjust its end time.
6. Add a 1-second karaoke lead-in to every line.
7. Go through each line in karaoke mode and merge/adjust syllable timings as needed.

<img src="https://github.com/user-attachments/assets/614cd8ca-d471-447c-8596-4ac800d690cf" width="25%" >

## Sample

Using our new [NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn](https://huggingface.co/NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn) fine-tuned model (default):

- **Umamusume: Pretty Derby - Legend Changer** ([2026.10.0](https://github.com/Japan7/yohane/releases/tag/2026.10.0)):
  [Video](https://www.youtube.com/watch?v=o8PQexPl6Ns),
  [Output](<samples/ゲーム【ウマ娘 プリティーダービー】ライブ動画「Legend-Changer」ゲームサイズVer. [o8PQexPl6Ns].ass>)

Using vocal-remover and MMS-FA models (`--separator vocal-remover --forced-aligner mms-fa`):

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
- [NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn HF model](https://huggingface.co/NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn)
