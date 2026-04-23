from dataclasses import dataclass
from functools import cached_property
from typing import Iterable


@dataclass(frozen=True)
class YohaneFATokenizer:
    vocabulary: dict[str, int]
    unk_token: str
    pad_token: str

    @cached_property
    def unk_token_id(self) -> int:
        return self.vocabulary[self.unk_token]

    @cached_property
    def pad_token_id(self) -> int:
        return self.vocabulary[self.pad_token]

    def encode(self, value: str) -> list[int]:
        return [self.vocabulary.get(t, self.unk_token_id) for t in value]

    def __len__(self) -> int:
        return len(self.vocabulary)

    @classmethod
    def from_iterable(
        cls,
        tokens: Iterable[str],
        *,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
    ) -> "YohaneFATokenizer":
        vocabulary = {token: idx for idx, token in enumerate(sorted(set(tokens)))}
        vocabulary[unk_token] = len(vocabulary)
        vocabulary[pad_token] = len(vocabulary)
        return cls(vocabulary=vocabulary, unk_token=unk_token, pad_token=pad_token)

    @classmethod
    def from_dataset(
        cls,
        path: str,
        *,
        split: str,
        max_duration: int,
        load_from_cache_file: bool,
    ) -> "YohaneFATokenizer":
        from yohane_fa.dataset import load_dataset

        dataset = load_dataset(
            path,
            split=split,
            max_duration=max_duration,
            load_from_cache_file=load_from_cache_file,
        )

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
