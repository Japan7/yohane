from dataclasses import dataclass
from functools import cached_property

import regex as re


@dataclass
class Text:
    raw: str
    lines: list["Line"]

    @cached_property
    def transcript(self):
        return [syllable for line in self.lines for syllable in line.transcript]

    @classmethod
    def from_raw(cls, raw: str):
        lines = [Line.from_raw(line) for line in raw.splitlines()]
        lines = list(filter(None, lines))
        return cls(raw, lines)


@dataclass
class Line:
    raw: str
    normalized: str
    words: list["Word"]

    @cached_property
    def transcript(self):
        return [syllable for word in self.words for syllable in word.syllables]

    @classmethod
    def from_raw(cls, raw: str):
        normalized = normalize_uroman(raw)
        if not normalized:
            return None
        words = [Word.from_value(word) for word in normalized.split()]
        return cls(raw, normalized, words)


@dataclass
class Word:
    value: str
    syllables: list[str]

    @classmethod
    def from_value(cls, value: str):
        syllables = split_word(value)
        return cls(value, syllables)


def normalize_uroman(text: str):
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z'\n ])", " ", text)
    text = re.sub("\n[\n ]+", "\n", text)
    text = re.sub(" +", " ", text)
    return text.strip()


# https://docs.karaokes.moe/aegisub/auto-split.lua
MUGEN_RE = re.compile(
    r"(?i)(?:(?<=[^sc])(?=h))|(?:(?<=[^kstnhfmrwpbdgzcj])(?=y))|(?:(?<=[^t])(?=s))|(?:(?=[ktnfmrwpbdgzcj]))|(?:(?<=[aeiou]|[^[:alnum:]])(?=[aeiou]))"
)


def split_word(word: str):
    splitter_str, _ = MUGEN_RE.subn("#@", word)
    syllabes = re.split("#@", splitter_str, flags=re.MULTILINE)
    syllabes = list(filter(None, syllabes))
    return syllabes
