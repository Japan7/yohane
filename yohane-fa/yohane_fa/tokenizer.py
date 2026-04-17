from dataclasses import dataclass
from functools import cached_property
from typing import Iterable


@dataclass(frozen=True)
class YohaneFATokenizer:
    vocabulary: dict[str, int]
    blank_token: str
    unk_token: str
    pad_token: str

    @cached_property
    def blank_id(self) -> int:
        return self.vocabulary[self.blank_token]

    @cached_property
    def unk_id(self) -> int:
        return self.vocabulary[self.unk_token]

    @cached_property
    def pad_id(self) -> int:
        return self.vocabulary[self.pad_token]

    def encode(self, token: str) -> int:
        return self.vocabulary.get(
            token if token != " " else self.blank_token,
            self.unk_id,
        )

    @classmethod
    def from_iterable(
        cls,
        tokens: Iterable[str],
        *,
        blank_token: str = "|",
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
    ) -> "YohaneFATokenizer":
        vocabulary = {token: idx for idx, token in enumerate(sorted(set(tokens)))}
        if " " in vocabulary:
            vocabulary[blank_token] = vocabulary[" "]
            del vocabulary[" "]
        else:
            vocabulary[blank_token] = len(vocabulary)
        vocabulary[unk_token] = len(vocabulary)
        vocabulary[pad_token] = len(vocabulary)
        return cls(vocabulary, blank_token, unk_token, pad_token)

    @classmethod
    def from_dataset(cls, path: str, *, split: str) -> "YohaneFATokenizer":
        from datasets import load_dataset

        dataset = load_dataset(path, split=split)
        vocab = dataset.map(
            lambda batch: {
                "vocab": list(
                    set(
                        letter
                        for sample in batch
                        for mora in sample
                        for letter in mora["value"]
                    )
                )
            },
            input_columns=["morae"],
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=None,
            keep_in_memory=True,
        )["vocab"]

        return cls.from_iterable(vocab)
