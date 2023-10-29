from importlib.metadata import metadata

from pysubs2 import SSAEvent, SSAFile
from torch import Tensor
from torchaudio.functional import TokenSpan

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

    for line in lyrics.lines:
        event: SSAEvent | None = None
        k_cumul = 0

        for word in line.words:
            spans = next(token_spans_iter)
            span_idx = 0

            for syllable in word.syllables:
                syllable_tokens = tokenizer([syllable])
                nb_tokens = len(syllable_tokens[0])

                # start and end time of syllable
                x0 = ratio * spans[span_idx].start
                x1 = ratio * spans[span_idx + nb_tokens - 1].end
                t_start = x0 / sample_rate  # s
                t_end = x1 / sample_rate  # s

                if event is None:
                    # new line, new event
                    event = SSAEvent(int(t_start * 1000))  # ms
                elif span_idx == 0:
                    # k tag logic:
                    # new word starting on the same line
                    # add a space and adjust timing
                    space_dt = round(t_start * 100 - event.start / 10 - k_cumul)  # cs
                    event.text += rf"{{\k{space_dt}}} "
                    k_cumul += space_dt  # cs

                # k tag logic:
                # snap the token end time to the next token start time
                try:
                    next_x0 = ratio * spans[span_idx + nb_tokens].start
                    next_t_start = next_x0 / sample_rate  # s
                    adjusted_t_end = next_t_start  # s
                except IndexError:
                    adjusted_t_end = t_end  # s

                k_duration = round((adjusted_t_end - t_start) * 100)  # cs
                k_cumul += k_duration  # cs

                event.text += rf"{{\k{k_duration}}}{syllable}"

                # increment event end time
                event.end = int(adjusted_t_end * 1000)  # ms

                # consumed tokens
                span_idx += nb_tokens

        if event is not None:
            # save the raw line in a comment
            comment = SSAEvent(event.start, event.end, line.raw, type="Comment")
            subs.append(comment)

            # save the timed event
            event.text = event.text.strip()
            subs.append(event)

    return subs
