import logging
from importlib.resources import as_file, files
from pathlib import Path
from typing import cast

import torch
import torchaudio
import vocal_remover.models
from torchaudio.pipelines import MMS_FA as bundle
from vocal_remover.inference import Separator
from vocal_remover.lib import nets, spec_utils

logger = logging.getLogger(__name__)


def prepare_audio(audio_file: str | Path, extract_vocals: bool):
    waveform, sample_rate = torchaudio.load(audio_file)  # type: ignore
    if extract_vocals:
        waveform = separate_vocals(waveform)
        torchaudio.save(audio_file.with_suffix(".vocals.wav"), waveform, sample_rate)  # type: ignore
    waveform = waveform.mean(0, keepdim=True)
    waveform = torchaudio.functional.resample(
        waveform, sample_rate, int(bundle.sample_rate)
    )
    return waveform


def separate_vocals(waveform: torch.Tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device=}")

    state_resource = files(vocal_remover.models) / "baseline.pth"
    with as_file(state_resource) as path:
        state = torch.load(path, map_location=device)

    separator_model = nets.CascadedNet(2048, 32, 128)
    separator_model.load_state_dict(state)
    separator_model.to(device)

    if waveform.ndim == 1:
        waveform = waveform.repeat(2, 1)
    waveform_spec = spec_utils.wave_to_spectrogram(waveform, 1024, 2048)

    sp = Separator(separator_model, device, 4, 256)
    _, vocals_spec = sp.separate(waveform_spec)

    vocals = spec_utils.spectrogram_to_wave(vocals_spec)
    return torch.Tensor(vocals)


def compute_alignments(waveform: torch.Tensor, transcript: list[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device=}")

    model = bundle.get_model()
    model.to(device)

    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        emission = cast(torch.Tensor, emission)
        tokens = tokenizer(transcript)
        tokens = cast(list[list[int]], tokens)
        token_spans = aligner(emission[0], tokens)

    return emission, token_spans
