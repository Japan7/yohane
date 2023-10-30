import logging
from abc import ABC, abstractmethod
from importlib.resources import as_file, files
from typing import cast

import torch
import torchaudio
import vocal_remover.models
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.pipelines import MMS_FA as fa_bundle
from torchaudio.transforms import Fade
from vocal_remover.inference import Separator
from vocal_remover.lib import nets, spec_utils

logger = logging.getLogger(__name__)


def compute_alignments(waveform: torch.Tensor, sample_rate: int, transcript: list[str]):
    """
    https://pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device=}")

    waveform = waveform.mean(0, keepdim=True)
    waveform, sample_rate = (
        torchaudio.functional.resample(
            waveform, sample_rate, int(fa_bundle.sample_rate)
        ),
        int(fa_bundle.sample_rate),
    )

    model = fa_bundle.get_model()
    model.to(device)

    tokenizer = fa_bundle.get_tokenizer()
    aligner = fa_bundle.get_aligner()

    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        emission = cast(torch.Tensor, emission)
        tokens = tokenizer(transcript)
        tokens = cast(list[list[int]], tokens)
        token_spans = aligner(emission[0], tokens)

    return emission, token_spans


class VocalsExtractor(ABC):
    @abstractmethod
    def __call__(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, int]:
        ...


class VocalRemoverVocalsExtractor(VocalsExtractor):
    """
    https://github.com/tsurumeso/vocal-remover
    """

    def __call__(self, waveform: torch.Tensor, sample_rate: int):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using {device=}")

        state_resource = files(vocal_remover.models) / "baseline.pth"
        with as_file(state_resource) as path:
            state = torch.load(path, map_location=device)

        separator_model = nets.CascadedNet(2048, 32, 128)
        separator_model.load_state_dict(state)
        separator_model.to(device)

        waveform = waveform.repeat(2, 1) if waveform.ndim == 1 else waveform

        waveform_spec = spec_utils.wave_to_spectrogram(waveform, 1024, 2048)

        sp = Separator(separator_model, device, 4, 256)
        _, vocals_spec = sp.separate(waveform_spec)

        vocals = spec_utils.spectrogram_to_wave(vocals_spec)

        return torch.Tensor(vocals), sample_rate


class HybridDemucsVocalsExtractor(VocalsExtractor):
    """
    https://pytorch.org/audio/2.1.0/tutorials/hybrid_demucs_tutorial.html
    """

    def __init__(self, segment=10.0, overlap=0.1):
        self.bundle = HDEMUCS_HIGH_MUSDB_PLUS
        self.segment = segment
        self.overlap = overlap

    def separate_sources(
        self,
        mix: torch.Tensor,
        sample_rate: int,
        model: torch.nn.Module,
        device: torch.device,
    ):
        batch, channels, length = mix.shape

        chunk_len = int(sample_rate * self.segment * (1 + self.overlap))
        start = 0
        end = chunk_len
        overlap_frames = self.overlap * sample_rate
        fade = Fade(
            fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear"
        )

        final = torch.zeros(batch, len(model.sources), channels, length, device=device)

        while start < length - overlap_frames:
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                out = model.forward(chunk)
            out = fade(out)
            final[:, :, :, start:end] += out
            if start == 0:
                fade.fade_in_len = int(overlap_frames)
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            end += chunk_len
            if end >= length:
                fade.fade_out_len = 0
        return final

    def __call__(self, waveform: torch.Tensor, sample_rate: int):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using {device=}")

        waveform, sample_rate = (
            torchaudio.functional.resample(
                waveform, sample_rate, self.bundle.sample_rate
            ),
            self.bundle.sample_rate,
        )
        waveform = waveform.to(device)

        model = self.bundle.get_model()
        model.to(device)

        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()  # normalization

        sources = self.separate_sources(waveform[None], sample_rate, model, device)[0]
        sources = sources * ref.std() + ref.mean()

        sources_list = model.sources
        sources = list(sources)

        audios = dict(zip(sources_list, sources))

        return audios["vocals"], sample_rate


VOCALS_EXTRACTORS: dict[str, type[VocalsExtractor]] = {
    "VocalRemover": VocalRemoverVocalsExtractor,
    "HybridDemucs": HybridDemucsVocalsExtractor,
}
