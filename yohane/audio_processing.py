from pathlib import Path
from typing import cast

import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle


def load_audio(audio_file: str | Path):
    waveform, sample_rate = torchaudio.load(audio_file)  # type: ignore
    waveform = waveform.mean(0, keepdim=True)
    waveform = torchaudio.functional.resample(
        waveform, sample_rate, int(bundle.sample_rate)
    )
    return waveform


def compute_alignments(waveform: torch.Tensor, transcript: list[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device=}")

    model = bundle.get_model()
    model.to(device)

    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        emission = cast(torch.Tensor, emission)
        tokens = cast(list[list[int]], tokenizer(transcript))
        token_spans = aligner(emission[0], tokens)

    return emission, token_spans
