from typing import Iterator

from pysubs2 import SSAEvent, SSAFile
from torch import Tensor
from torchaudio.functional import TokenSpan

from yohane.force_alignment import bundle
from yohane.text_processing import Line, Text, Word


def make_ass(
    lyrics_txt: Text,
    waveform: Tensor,
    emission: Tensor,
    token_spans: list[list[TokenSpan]],
):
    num_frames = emission.size(1)
    ratio = waveform.size(1) / num_frames
    sample_rate = bundle.sample_rate

    subs = SSAFile()
    event = SSAEvent()

    lines_iter = iter(lyrics_txt.lines)
    curr_line: Line
    words_iter: Iterator[Word] = iter(())
    curr_word: Word
    syllables_iter: Iterator[str] = iter(())
    curr_syllable: str

    for i, spans in enumerate(token_spans):
        x0 = ratio * spans[0].start
        x1 = ratio * spans[-1].end

        t_start = x0 / sample_rate
        t_end = x1 / sample_rate

        next_t_start = None
        if i < len(token_spans) - 1:
            next_x0 = ratio * token_spans[i + 1][0].start
            next_t_start = next_x0 / sample_rate

        try:
            curr_syllable = next(syllables_iter)
        except StopIteration:
            try:
                # New word
                curr_word = next(words_iter)
                syllables_iter = iter(curr_word.syllables)
                curr_syllable = next(syllables_iter)

                event.text += " "
            except StopIteration:
                try:
                    # New line
                    curr_line = next(lines_iter)
                    words_iter = iter(curr_line.words)
                    curr_word = next(words_iter)
                    syllables_iter = iter(curr_word.syllables)
                    curr_syllable = next(syllables_iter)

                    event.text = event.text.strip()
                    if event.text:
                        subs.append(event)
                    event = SSAEvent(int(t_start * 1000))
                except StopIteration:
                    raise RuntimeError("???")

        k_start = t_start
        k_end = next_t_start if next_t_start is not None else t_end
        event.text += rf"{{\k{round((k_end - k_start) * 100)}}}{curr_syllable}"
        event.end = int(t_end * 1000)

    return subs
