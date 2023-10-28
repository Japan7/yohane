import re
from dataclasses import dataclass
from functools import cached_property


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
        words = [Word.from_raw(word) for word in normalized.split()]
        return cls(raw, normalized, words)


@dataclass
class Word:
    raw: str
    syllables: list[str]

    @classmethod
    def from_raw(cls, raw: str):
        syllables = split_romaji_word(raw)
        return cls(raw, syllables)


def normalize_uroman(text: str):
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z'\n ])", " ", text)
    text = re.sub("\n[\n ]+", "\n", text)
    text = re.sub(" +", " ", text)
    return text.strip()


def split_romaji_word(word: str):
    """
    https://github.com/animefn/ksplitter/blob/51ea5abde8542703c8d89143304b69f11576b678/ksplitter.py#L44C1-L157C27
    """
    karaSplit_array = []
    ln = len(word)
    letter_index = 0
    while letter_index < ln:
        letter = word[letter_index]
        letter1 = ""
        if 0 <= letter_index + 1 < len(word):
            letter1 = word[letter_index + 1]

        if letter.lower() in "rymnhk":
            if letter1.lower() in "aeiouō":
                karaSplit_array.append(letter + letter1)
                letter_index = letter_index + 2
            else:
                karaSplit_array.append(letter)
                letter_index = letter_index + 1

        elif letter.lower() == "w":
            if letter1.lower() in "aoō":
                karaSplit_array.append(letter + letter1)
                letter_index = letter_index + 2
            else:
                karaSplit_array.append(letter)
                letter_index = letter_index + 1
        # if letter ==  "y":
        elif letter.lower() == "t":
            if letter1.lower() in "aeoō":
                karaSplit_array.append(letter + letter1)
                letter_index = letter_index + 2
            elif letter1.lower() == "s":
                if 0 <= letter_index + 2 < len(word):
                    letter2 = word[letter_index + 2]
                    karaSplit_array.append(letter + letter1 + letter2)
                letter_index = letter_index + 3
            else:
                karaSplit_array.append(letter)
                letter_index = letter_index + 1

        elif letter.lower() == "c":
            if letter1.lower() == "h":
                if 0 <= letter_index + 2 < len(word):
                    letter2 = word[letter_index + 2]
                    karaSplit_array.append(letter + letter1 + letter2)
                letter_index = letter_index + 3
            else:
                karaSplit_array.append(letter + letter1)
                letter_index = letter_index + 2

        elif letter.lower() == "s":
            if letter1.lower() in "aueoō":
                karaSplit_array.append(letter + letter1)
                letter_index = letter_index + 2
            elif letter1.lower() == "h":
                if 0 <= letter_index + 2 < len(word):
                    letter2 = word[letter_index + 2]
                    karaSplit_array.append(letter + letter1 + letter2)
                letter_index = letter_index + 3
            else:
                karaSplit_array.append(letter)
                letter_index = letter_index + 1
        elif letter.lower() == "f":
            if letter1 == "u":
                karaSplit_array.append(letter + letter1)
                letter_index = letter_index + 2
            else:
                karaSplit_array.append(letter)
                letter_index = letter_index + 1

        elif letter.lower() in "aeiou":
            if letter1.lower() in "aeiou":
                karaSplit_array.append(letter)
                letter_index = letter_index + 1
            else:
                # this should not happen it is only the case above, nothing is supposed to end with these letter
                karaSplit_array.append(letter)
                letter_index = letter_index + 1

        else:
            if letter1.lower() in "aeiou":
                karaSplit_array.append(letter + letter1)
                letter_index = letter_index + 2
            else:
                karaSplit_array.append(letter)
                letter_index = letter_index + 1

    return karaSplit_array
