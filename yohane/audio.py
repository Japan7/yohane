import logging
from abc import ABC, abstractmethod
from typing import Callable, cast

import torch
from torchaudio.functional import TokenSpan, resample
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS, MMS_FA
from torchaudio.pipelines._wav2vec2 import aligner
from torchaudio.transforms import Fade
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

logger = logging.getLogger(__name__)

TokenizerFn = Callable[[list[str]], list[list[int]]]


class ForcedAligner(ABC):
    @abstractmethod
    def tokenize(
        self,
        batch: list[str],
    ) -> list[list[int]]: ...

    @abstractmethod
    def align(
        self,
        tokens: list[list[int]],
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> tuple[torch.Tensor, list[list[TokenSpan]]]: ...


class TorchAudioForcedAligner(ForcedAligner):
    """
    https://pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html
    """

    bundle = MMS_FA

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = self.bundle.get_tokenizer()
        self.model = self.bundle.get_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.aligner = self.bundle.get_aligner()

    def tokenize(self, batch: list[str]):
        return cast(list[list[int]], self.tokenizer(batch))

    def align(self, tokens: list[list[int]], waveform: torch.Tensor, sample_rate: int):
        logger.info(f"TorchAudioForcedAligner: runs MMS_FA on {self.device=}")
        waveform = resample(waveform, sample_rate, int(self.bundle.sample_rate))
        waveform = waveform.mean(0, keepdim=True)
        with torch.inference_mode():
            emission, _ = self.model(waveform.to(self.device))
            emission = cast(torch.Tensor, emission)
        token_spans = self.aligner(emission[0], tokens)
        return emission, token_spans


class Wav2Vec2ForcedAligner(ForcedAligner):
    def __init__(self, model: str) -> None:
        super().__init__()
        self.model_name = model
        self.processor = Wav2Vec2Processor.from_pretrained(model)
        self.model = Wav2Vec2ForCTC.from_pretrained(model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # pyright: ignore[reportArgumentType]
        self.aligner = aligner.Aligner(blank=self.tokenizer.word_delimiter_token_id)

    @property
    def tokenizer(self) -> Wav2Vec2CTCTokenizer:
        return self.processor.tokenizer  # pyright: ignore[reportAttributeAccessIssue]

    def tokenize(self, batch: list[str]):
        return [self.tokenizer.encode(e, add_special_tokens=False) for e in batch]

    def align(self, tokens: list[list[int]], waveform: torch.Tensor, sample_rate: int):
        logger.info(f"Wav2Vec2ForcedAligner: runs {self.model_name} on {self.device=}")
        target_sample_rate = self.processor.feature_extractor.sampling_rate  # pyright: ignore[reportAttributeAccessIssue]
        waveform = resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
        waveform = waveform.mean(0)
        inputs = self.processor(
            audio=waveform.numpy(),
            sampling_rate=sample_rate,  # pyright: ignore[reportCallIssue]
            return_tensors="pt",  # pyright: ignore[reportCallIssue]
        )
        with torch.inference_mode():
            outputs = self.model(**inputs.to(self.device))
            emission = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        token_spans = self.aligner(emission[0], tokens)
        return emission, token_spans


class Separator(ABC):
    @abstractmethod
    def __call__(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, int]: ...


class VocalRemoverSeparator(Separator):
    """
    https://github.com/tsurumeso/vocal-remover
    """

    def __init__(self):
        super().__init__()
        from vocal_remover.transformer.modeling import VocalRemoverModel
        from vocal_remover.transformer.pipeline import VocalRemoverPipeline

        self.model = VocalRemoverModel.from_pretrained(
            "NextFire/tsurumeso-vocal-remover"
        )
        self.pipeline = VocalRemoverPipeline(self.model)

    def __call__(self, waveform: torch.Tensor, sample_rate: int):
        outputs = self.pipeline(waveform)
        return torch.Tensor(outputs["vocals"]), sample_rate


class HybridDemucsSeparator(Separator):
    """
    https://pytorch.org/audio/2.1.0/tutorials/hybrid_demucs_tutorial.html
    """

    bundle = HDEMUCS_HIGH_MUSDB_PLUS

    def __init__(self, segment=10.0, overlap=0.1):
        super().__init__()
        self.segment = segment
        self.overlap = overlap
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def separate_sources(
        self,
        mix: torch.Tensor,
        sample_rate: int,
        model: torch.nn.Module,
    ):
        batch, channels, length = mix.shape

        chunk_len = int(sample_rate * self.segment * (1 + self.overlap))
        start = 0
        end = chunk_len
        overlap_frames = self.overlap * sample_rate
        fade = Fade(
            fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear"
        )

        sources_list = cast(list[str], model.sources)
        final = torch.zeros(
            batch, len(sources_list), channels, length, device=self.device
        )

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
        logger.info(f"HybridDemucsSeparator: runs on {self.device=}")

        waveform, sample_rate = (
            resample(waveform, sample_rate, self.bundle.sample_rate),
            self.bundle.sample_rate,
        )
        waveform = waveform.to(self.device)

        model = self.bundle.get_model()
        model.to(self.device)

        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()  # normalization

        sources = self.separate_sources(waveform[None], sample_rate, model)[0]
        sources = sources * ref.std() + ref.mean()

        sources_list = cast(list[str], model.sources)
        sources = list(sources)

        audios = dict(zip(sources_list, sources))

        return audios["vocals"], sample_rate
