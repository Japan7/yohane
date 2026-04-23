from typing import Any, TypedDict, cast

import lightning as L
import torch
import torchaudio
from datasets import Dataset
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchcodec.decoders import AudioDecoder

from yohane_fa.dataset import load_dataset
from yohane_fa.modules import YohaneFA
from yohane_fa.tokenizer import YohaneFATokenizer


class DatasetMora(TypedDict):
    value: str
    start: int
    end: int


class DatasetRow(TypedDict):
    audio: AudioDecoder
    morae: list[DatasetMora]


class ModelInput(TypedDict):
    input_values: torch.Tensor
    labels: torch.Tensor


class KaraokeAlignementsDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        dataset_path: str,
        dataset_split: str = "train",
        dataset_max_duration: int = 150,
        mel_target_sample_rate: int = 16000,
        mel_n_fft: int = 400,
        mel_hop_length: int = 160,
        mel_n_mels: int = 80,
        load_from_cache_file: bool = True,
        batch_size: int = 1,
        dampening_factor: float = 0.1,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.dataset_max_duration = dataset_max_duration
        self.mel_target_sample_rate = mel_target_sample_rate
        self.mel_n_fft = mel_n_fft
        self.mel_hop_length = mel_hop_length
        self.mel_n_mels = mel_n_mels
        self.load_from_cache_file = load_from_cache_file
        self.batch_size = batch_size
        self.dampening_factor = dampening_factor
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.mel_target_sample_rate,
            n_fft=self.mel_n_fft,
            hop_length=self.mel_hop_length,
            n_mels=self.mel_n_mels,
        )
        self.tokenizer = YohaneFATokenizer.from_dataset(
            self.dataset_path,
            split=self.dataset_split,
            max_duration=self.dataset_max_duration,
            load_from_cache_file=self.load_from_cache_file,
        )
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if all((self.train_dataset, self.val_dataset, self.test_dataset)):
            return
        dataset = load_dataset(
            self.dataset_path,
            split=self.dataset_split,
            max_duration=self.dataset_max_duration,
            load_from_cache_file=self.load_from_cache_file,
        )
        dataset = dataset.map(
            self._prepare_example,
            remove_columns=dataset.column_names,
            new_fingerprint="yohane_fa_prepared",
            load_from_cache_file=self.load_from_cache_file,
        )
        split = dataset.train_test_split(test_size=0.15)
        self.train_dataset = split["train"]
        split = split["test"].train_test_split(test_size=0.5)
        self.val_dataset = split["train"]
        self.test_dataset = split["test"]

    def train_dataloader(self) -> Any:
        assert self.train_dataset
        return DataLoader(
            self.train_dataset.with_format("torch"),  # pyright: ignore[reportArgumentType]
            batch_size=self.batch_size,
            collate_fn=self._collate_batch,
        )

    def val_dataloader(self) -> Any:
        assert self.val_dataset
        return DataLoader(
            self.val_dataset.with_format("torch"),  # pyright: ignore[reportArgumentType]
            batch_size=self.batch_size,
            collate_fn=self._collate_batch,
        )

    def test_dataloader(self) -> Any:
        assert self.test_dataset
        return DataLoader(
            self.test_dataset.with_format("torch"),  # pyright: ignore[reportArgumentType]
            batch_size=self.batch_size,
            collate_fn=self._collate_batch,
        )

    def _prepare_example(self, example: DatasetRow) -> ModelInput:
        waveform = example["audio"].get_all_samples().data
        duration_seconds = example["audio"].metadata.duration_seconds
        assert duration_seconds is not None
        input_values = self.mel_transform(waveform).squeeze(0).transpose(0, 1)
        input_values = torch.log(input_values.clamp_min(1e-5))
        labels = self._build_labels(
            example["morae"],
            float(duration_seconds * 1000),
            input_values.size(0),
        )
        return {"input_values": input_values, "labels": labels}

    def _build_labels(
        self,
        morae: list[DatasetMora],
        audio_duration: float,
        input_length: int,
    ) -> torch.Tensor:
        frames_per_ms = input_length / audio_duration
        labels = torch.full((input_length,), self.tokenizer.pad_token_id)
        for mora in morae:
            value = mora["value"]
            if not value.strip():
                continue
            start_frame = round(mora["start"] * frames_per_ms)
            assert start_frame < input_length
            end_frame = round(mora["end"] * frames_per_ms)
            assert start_frame <= end_frame <= input_length
            n_frames = end_frame - start_frame
            input_ids = self.tokenizer.encode(value)
            for idx, frame in enumerate(range(start_frame, end_frame)):
                if labels[frame] == self.tokenizer.pad_token_id:
                    token_idx = (idx * len(input_ids)) // n_frames
                    labels[frame] = input_ids[token_idx]
                else:
                    # Mask overlapping alignments with -100 to ignore them in loss calculation
                    labels[frame] = -100
        return labels

    def _collate_batch(self, batch: list[ModelInput]) -> ModelInput:
        input_values = [example["input_values"] for example in batch]
        labels = [example["labels"] for example in batch]
        padded_input_values = pad_sequence(
            input_values,
            batch_first=True,
            padding_value=0.0,
        )
        padded_labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        return {"input_values": padded_input_values, "labels": padded_labels}


class YohaneFALightning(L.LightningModule):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim_multiplier: int,
        output_dim: int,
        pad_token_id: int,
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = YohaneFA(
            input_dim,
            input_dim * hidden_dim_multiplier,
            output_dim,
            dropout=dropout,
        )
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: ModelInput, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch: ModelInput, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch: ModelInput, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, stage="test")

    def _shared_step(
        self,
        batch: ModelInput,
        batch_idx: int,
        *,
        stage: str,
    ) -> torch.Tensor:
        logits = cast(torch.Tensor, self(batch["input_values"]))
        labels = batch["labels"]
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            valid_mask = labels != -100
            frame_accuracy = self._masked_accuracy(predictions, labels, valid_mask)
            char_mask = valid_mask & (labels != self.pad_token_id)
            char_accuracy = self._masked_accuracy(predictions, labels, char_mask)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_frame_accuracy", frame_accuracy, prog_bar=True)
        self.log(f"{stage}_char_accuracy", char_accuracy)
        return loss

    @staticmethod
    def _masked_accuracy(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.any(mask):
            return predictions.new_tensor(0)
        correct = (predictions[mask] == labels[mask]).float()
        return correct.mean()
