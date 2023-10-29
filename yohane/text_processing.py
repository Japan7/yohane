from dataclasses import dataclass
from functools import cached_property

import regex as re


@dataclass
class Text:
    raw: str

    @cached_property
    def normalized(self):
        return normalize_uroman(self.raw)

    @cached_property
    def transcript(self):
        return self.normalized.split()


@dataclass
class Lyrics(Text):
    @cached_property
    def lines(self):
        return [Line(line) for line in filter(None, self.raw.splitlines())]


@dataclass
class Line(Text):
    @cached_property
    def words(self):
        return [Word(word) for word in filter(None, self.transcript)]


@dataclass
class Word(Text):
    @cached_property
    def syllables(self):
        return syllables_splitter(self.normalized)


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


def syllables_splitter(word: str):
    splitter_str, _ = MUGEN_RE.subn("#@", word)
    syllabes = re.split("#@", splitter_str, flags=re.MULTILINE)
    syllabes = list(filter(None, syllabes))
    return syllabes
