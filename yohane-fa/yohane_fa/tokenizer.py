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
