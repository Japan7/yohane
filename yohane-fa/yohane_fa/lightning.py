from functools import cached_property
from typing import Any, TypedDict

import lightning as L
import torch
import torchaudio
from datasets import Dataset, load_dataset
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchcodec.decoders import AudioDecoder

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
        dataset_max_duration: int = 120,
        mel_target_sample_rate: int = 16000,
        mel_n_fft: int = 400,
        mel_hop_length: int = 160,
        mel_n_mels: int = 80,
        batch_size: int = 1,
        num_workers: int = 1,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_split = dataset_split
        self.dataset_max_duration = dataset_max_duration
        self.mel_target_sample_rate = mel_target_sample_rate
        self.mel_n_fft = mel_n_fft
        self.mel_hop_length = mel_hop_length
        self.mel_n_mels = mel_n_mels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = YohaneFATokenizer.from_dataset(
            self.dataset_path,
            split=self.dataset_split,
        )
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    @cached_property
    def ms_per_frame(self) -> float:
        return self.mel_hop_length / self.mel_target_sample_rate * 1000

    def setup(self, stage: str | None = None) -> None:
        if all((self.train_dataset, self.val_dataset, self.test_dataset)):
            return

        dataset = load_dataset(self.dataset_path, split=self.dataset_split)
        dataset = dataset.filter(
            lambda x: x.metadata.duration_seconds <= self.dataset_max_duration,
            input_columns=["audio"],
            num_proc=self.num_workers,
        )

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.mel_target_sample_rate,
            n_fft=self.mel_n_fft,
            hop_length=self.mel_hop_length,
            n_mels=self.mel_n_mels,
        )
        dataset = dataset.map(
            self._prepare_example,
            fn_kwargs={"mel_transform": mel_transform},
            remove_columns=dataset.column_names,
            num_proc=self.num_workers,
            new_fingerprint="yohane_fa_prepared_dataset",
        )

        split = dataset.train_test_split(test_size=0.1)
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
            num_workers=self.num_workers,
            persistent_workers=True,
            multiprocessing_context=(
                "fork" if torch.backends.mps.is_available() else None
            ),
        )

    def val_dataloader(self) -> Any:
        assert self.val_dataset
        return DataLoader(
            self.val_dataset.with_format("torch"),  # pyright: ignore[reportArgumentType]
            batch_size=self.batch_size,
            collate_fn=self._collate_batch,
            num_workers=self.num_workers,
            persistent_workers=True,
            multiprocessing_context=(
                "fork" if torch.backends.mps.is_available() else None
            ),
        )

    def test_dataloader(self) -> Any:
        assert self.test_dataset
        return DataLoader(
            self.test_dataset.with_format("torch"),  # pyright: ignore[reportArgumentType]
            batch_size=self.batch_size,
            collate_fn=self._collate_batch,
            num_workers=self.num_workers,
            persistent_workers=True,
            multiprocessing_context=(
                "fork" if torch.backends.mps.is_available() else None
            ),
        )

    def _prepare_example(
        self,
        example: DatasetRow,
        *,
        mel_transform: torchaudio.transforms.MelSpectrogram,
    ) -> ModelInput:
        waveform = example["audio"].get_all_samples().data
        input_values = mel_transform(waveform).squeeze(0).transpose(0, 1)
        input_values = torch.log(input_values.clamp_min(1e-5))
        labels = self._build_labels(example["morae"], input_values.size(0))
        return {"input_values": input_values, "labels": labels}

    def _build_labels(
        self,
        morae: list[DatasetMora],
        input_length: int,
    ) -> torch.Tensor:
        labels = torch.full((input_length,), self.tokenizer.blank_id)
        for mora in morae:
            characters = list(mora["value"])
            start_frame = round(mora["start"] / self.ms_per_frame)
            if start_frame >= input_length:
                continue
            end_frame = min(round(mora["end"] / self.ms_per_frame), input_length)
            if end_frame <= start_frame:
                continue
            for frame in range(start_frame, end_frame):
                position = (frame - start_frame) / (end_frame - start_frame)
                char_idx = int(position * len(characters))
                labels[frame] = self.tokenizer.encode(characters[char_idx])
        return labels

    def _collate_batch(self, batch: list[ModelInput]) -> ModelInput:
        input_values = [example["input_values"] for example in batch]
        labels = [example["labels"] for example in batch]
        padded_input_values = pad_sequence(
            input_values,
            batch_first=True,
        )
        padded_labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.tokenizer.pad_id,
        )
        return {"input_values": padded_input_values, "labels": padded_labels}


class YohaneFALightning(L.LightningModule):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        blank_token_id: int,
        pad_token_id: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = YohaneFA(input_dim, hidden_dim, output_dim)
        self.blank_token_id = blank_token_id
        self.pad_token_id = pad_token_id

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
        logits = self(batch["input_values"])
        labels = batch["labels"]

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=self.pad_token_id,
        )

        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            valid_mask = labels != self.pad_token_id
            frame_accuracy = self._masked_accuracy(predictions, labels, valid_mask)
            char_mask = valid_mask & (labels != self.blank_token_id)
            char_accuracy = self._masked_accuracy(predictions, labels, char_mask)

        self.log(f"{stage}_loss", loss, prog_bar=stage != "test")
        self.log(f"{stage}_frame_accuracy", frame_accuracy, prog_bar=stage != "test")
        self.log(f"{stage}_char_accuracy", char_accuracy, prog_bar=False)

        return loss

    @staticmethod
    def _masked_accuracy(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.any(mask):
            return torch.tensor(0.0)
        correct = (predictions[mask] == labels[mask]).float()
        return correct.mean()
