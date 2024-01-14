from dataclasses import dataclass
from functools import cached_property

import regex as re
import pyphen

@dataclass
class _Text:
    raw: str
    language: str
    @cached_property
    def normalized(self):
        return normalize_uroman(self.raw)

    @cached_property
    def transcript(self):
        return self.normalized.split()


@dataclass
class Lyrics(_Text):
    @cached_property
    def lines(self):
        return [Line(line, self.language) for line in filter(None, self.raw.splitlines())]


@dataclass
class Line(_Text):
    @cached_property
    def words(self):
        return [Word(word, self.language) for word in filter(None, self.transcript)]


@dataclass
class Word(_Text):
    @cached_property
    def syllables(self):
        res = self.normalized
        if self.language == 'jp':
            res = auto_split(res)
        else:
            if self.language == 'en':
                dic = pyphen.Pyphen(lang="en_EN")
            elif self.language == 'fr':
                dic = pyphen.Pyphen(lang="fr_FR")
            else:
                raise ValueError(f"Unsupported language {self.language}")
            res = dic.inserted(res)
            res = res.split("-")
        return res


def normalize_uroman(text: str):
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z'\n ])", " ", text)
    text = re.sub("\n[\n ]+", "\n", text)
    text = re.sub(" +", " ", text)
    return text.strip()


# https://docs.karaokes.moe/aegisub/auto-split.lua
AUTO_SPLIT_RE = re.compile(
    r"(?i)(?:(?<=[^sc])(?=h))|(?:(?<=[^kstnhfmrwpbdgzcj])(?=y))|(?:(?<=[^t])(?=s))|(?:(?=[ktnfmrwpbdgzcj]))|(?:(?<=[aeiou]|[^[:alnum:]])(?=[aeiou]))"
)


def auto_split(word: str):
    splitter_str, _ = AUTO_SPLIT_RE.subn("#@", word)
    syllables = re.split("#@", splitter_str, flags=re.MULTILINE)
    syllables = list(filter(None, syllables))
    return syllables
