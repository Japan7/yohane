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


def make_ass(
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

    subs = SSAFile()
    subs.info["Original Timing"] = IDENTIFIER

    token_spans_iter = iter(token_spans)
    add_syllable = partial(_add_syllable, ratio, sample_rate, tokenizer)

    for line in lyrics.lines:
        event: SSAEvent | None = None
        k_cumul = 0

        for word in line.words:
            spans = next(token_spans_iter)
            span_idx = 0
            add_word_syllable = partial(add_syllable, spans)

            for syllable in word.syllables:
                event, nb_tokens, added_k = add_word_syllable(
                    event, syllable, span_idx, k_cumul
                )
                span_idx += nb_tokens
                k_cumul += added_k

        if event is not None:
            # save the raw line in a comment
            comment = SSAEvent(event.start, event.end, line.raw, type="Comment")
            subs.extend((comment, event))

    return subs


def _add_syllable(
    ratio: float,
    sample_rate: float,
    tokenizer: Wav2Vec2FABundle.Tokenizer,
    spans: list[TokenSpan],
    event: SSAEvent | None,
    syllable: str,
    span_idx: int,
    k_cumul: int,
):
    syllable_tokens = tokenizer([syllable])
    nb_tokens = len(syllable_tokens[0])

    # start and end time of syllable
    x0 = ratio * spans[span_idx].start
    x1 = ratio * spans[span_idx + nb_tokens - 1].end
    t_start = x0 / sample_rate  # s
    t_end = x1 / sample_rate  # s

    added_k = 0  # cs

    if event is None:
        # new line, new event
        event = SSAEvent(int(t_start * 1000))  # ms
    elif span_idx == 0:
        # k tag logic:
        # new word starting on the same line
        # add a space and adjust timing
        space_dt = round(t_start * 100 - event.start / 10 - k_cumul)  # cs
        event.text += rf"{{\k{space_dt}}} "
        added_k += space_dt  # cs

    # k tag logic:
    # snap the token end time to the next token start time
    try:
        next_x0 = ratio * spans[span_idx + nb_tokens].start
        next_t_start = next_x0 / sample_rate  # s
        adjusted_t_end = next_t_start  # s
    except IndexError:
        adjusted_t_end = t_end  # s

    k_duration = round((adjusted_t_end - t_start) * 100)  # cs
    added_k += k_duration  # cs

    event.text += rf"{{\k{k_duration}}}{syllable}"

    # increment event end time
    event.end = int(adjusted_t_end * 1000)  # ms

    return event, nb_tokens, added_k
