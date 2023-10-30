from dataclasses import dataclass
from functools import partial
from importlib.metadata import metadata

from pysubs2 import SSAEvent, SSAFile
from torch import Tensor
from torchaudio.functional import TokenSpan
from torchaudio.pipelines import Wav2Vec2FABundle

from yohane.audio import bundle
from yohane.lyrics import Lyrics

PKG_META = metadata(__package__)
IDENTIFIER = f"{PKG_META['Name']} {PKG_META['Version']} ({PKG_META['Home-page']})"


@dataclass
class TimedSyllable:
    value: str
    start_s: float  # s
    end_s: float  # s

    def k_duration(self, snap_to: float | None = None):
        k_s = (snap_to if snap_to is not None else self.end_s) - self.start_s  # s
        return round(k_s * 100)  # cs


def make_ass(
    lyrics: Lyrics,
    waveform: Tensor,
    emission: Tensor,
    token_spans: list[list[TokenSpan]],
):
    all_line_syllables = time_lyrics(lyrics, waveform, emission, token_spans)

    subs = SSAFile()
    subs.info["Original Timing"] = IDENTIFIER

    for line, syllables in zip(lyrics.lines, all_line_syllables):
        start_syl = syllables[0]
        end_syl = syllables[-1]
        assert start_syl is not None and end_syl is not None

        event = SSAEvent(round(start_syl.start_s * 1000), round(end_syl.end_s * 1000))

        for i, syllable in enumerate(syllables):
            if syllable is None:  # space
                continue

            value = syllable.value

            snap_to_i = None
            if i < len(syllables) - 1:  # not last syllable in line
                if syllables[i + 1] is None:  # next is space
                    value += " "
                    snap_to_i = i + 2  # snap to syllable after the space
                else:
                    snap_to_i = i + 1  # snap to next syllable

            if snap_to_i is not None:
                snap_to_syl = syllables[snap_to_i]
                assert snap_to_syl is not None
                snap_to = snap_to_syl.start_s
            else:
                snap_to = None

            k_duration = syllable.k_duration(snap_to=snap_to)  # cs
            event.text += rf"{{\k{k_duration}}}{value}"

        # save the raw line in a comment
        comment = SSAEvent(event.start, event.end, line.raw, type="Comment")
        subs.extend((comment, event))

    return subs


def time_lyrics(
    lyrics: Lyrics,
    waveform: Tensor,
    emission: Tensor,
    token_spans: list[list[TokenSpan]],
):
    # audio processing parameters
    num_frames = emission.size(1)
    ratio = waveform.size(1) / num_frames
    sample_rate = bundle.sample_rate
    tokenizer = bundle.get_tokenizer()

    token_spans_iter = iter(token_spans)
    add_syllable = partial(_time_syllable, ratio, sample_rate, tokenizer)

    all_line_syllables: list[list[TimedSyllable | None]] = []

    for line in lyrics.lines:
        line_syllables: list[TimedSyllable | None] = []

        for word in line.words:
            spans = next(token_spans_iter)
            span_idx = 0
            time_syllable = partial(add_syllable, spans)

            for syllable in word.syllables:
                nb_tokens, timed_syllable = time_syllable(syllable, span_idx)
                span_idx += nb_tokens
                line_syllables.append(timed_syllable)

            line_syllables.append(None)

        line_syllables = line_syllables[:-1]  # remove trailing None
        all_line_syllables.append(line_syllables)

    try:
        next(token_spans_iter)  # make sure we used all spans
        raise RuntimeError("not all spans were used")
    except StopIteration:
        pass

    return all_line_syllables


def _time_syllable(
    ratio: float,
    sample_rate: float,
    tokenizer: Wav2Vec2FABundle.Tokenizer,
    spans: list[TokenSpan],
    syllable: str,
    span_idx: int,
):
    syllable_tokens = tokenizer([syllable])
    nb_tokens = len(syllable_tokens[0])

    # start and end time of syllable
    x0 = ratio * spans[span_idx].start
    x1 = ratio * spans[span_idx + nb_tokens - 1].end
    t_start = x0 / sample_rate  # s
    t_end = x1 / sample_rate  # s

    timed_syllable = TimedSyllable(syllable, t_start, t_end)

    return nb_tokens, timed_syllable
