import json
from pathlib import Path

import torch
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from yohane_fa.lightning import KaraokeAlignementsDataModule, YohaneFALightning


class YohaneFALightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "data.mel_n_mels",
            "model.input_dim",
        )
        parser.link_arguments(
            "data",
            "model.output_dim",
            compute_fn=lambda data: len(data.tokenizer.vocabulary),
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data",
            "model.blank_token_id",
            compute_fn=lambda data: data.tokenizer.blank_id,
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data",
            "model.pad_token_id",
            compute_fn=lambda data: data.tokenizer.pad_id,
            apply_on="instantiate",
        )

    def before_fit(self) -> None:
        assert self.datamodule.tokenizer
        vocab_json = Path(self.trainer.log_dir or ".") / "vocab.json"
        vocab_json.parent.mkdir(parents=True, exist_ok=True)
        vocab_json.write_text(
            json.dumps(self.datamodule.tokenizer.vocabulary, indent=2) + "\n"
        )


def cli_main():
    torch.set_float32_matmul_precision("high")
    _ = YohaneFALightningCLI(YohaneFALightning, KaraokeAlignementsDataModule)


if __name__ == "__main__":
    cli_main()
