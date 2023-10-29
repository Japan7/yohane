from typing import cast
from pysubs2 import SSAEvent, SSAFile
from torch import Tensor
from torchaudio.functional import TokenSpan

from yohane.audio_processing import bundle
from yohane.text_processing import Lyrics


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

    # init subs and event
    subs = SSAFile()
    event = SSAEvent(-1)
    k_cumul = 0

    # init iterators
    lines_iter = iter(lyrics.lines)
    curr_line = next(lines_iter)
    words_iter = iter(curr_line.words)
    curr_word = next(words_iter)
    syllables_iter = iter(curr_word.syllables)

    # iterate over all aligned tokens
    # 1 span = 1 word
    for i, spans in enumerate(token_spans):
        j = 0
        while True:
            curr_syllable = next(syllables_iter)
            syllable_tokens = cast(list[list[int]], tokenizer([curr_syllable]))
            nb_tokens = len(syllable_tokens[0])

            # start and end time of syllable
            x0 = ratio * spans[j].start
            x1 = ratio * spans[j + nb_tokens - 1].end
            t_start = x0 / sample_rate  # s
            t_end = x1 / sample_rate  # s

            if event.start == -1:
                # set event start time if this is a new one
                event.start = int(t_start * 1000)  # ms
            elif j == 0:
                # k tag logic:
                # if this is a new word, add a space and adjust timing
                space_duration = round(t_start * 100 - event.start / 10 - k_cumul)  # cs
                event.text += rf"{{\k{space_duration}}} "
                k_cumul += space_duration  # cs

            # k tag logic:
            # snap the token end time to the next token start time
            try:
                next_x0 = ratio * spans[j + nb_tokens].start
                next_t_start = next_x0 / sample_rate
                adjusted_t_end = next_t_start
            except IndexError:
                adjusted_t_end = t_end

            k_duration = round((adjusted_t_end - t_start) * 100)  # cs
            k_cumul += k_duration  # cs

            event.text += rf"{{\k{k_duration}}}{curr_syllable}"

            # increment event end time
            event.end = int(adjusted_t_end * 1000)  # ms

            j += nb_tokens
            if j >= len(spans):
                break

        # if that was the last token, don't enter the iter logic
        if i == len(token_spans) - 1:
            break

        try:
            # add a space, advance to next word
            curr_word = next(words_iter)
            syllables_iter = iter(curr_word.syllables)
        except StopIteration:
            try:
                # no more words in line:
                # save the event in subs, reset it and advance to next line

                # save the raw line in a comment
                comment = SSAEvent(
                    event.start, event.end, curr_line.raw, type="Comment"
                )
                subs.append(comment)

                # save the timed event
                event.text = event.text.strip()
                subs.append(event)
                event = SSAEvent(-1)
                k_cumul = 0

                # iterators logic
                curr_line = next(lines_iter)
                words_iter = iter(curr_line.words)
                curr_word = next(words_iter)
                syllables_iter = iter(curr_word.syllables)
            except StopIteration as e:
                raise RuntimeError("should not happen") from e

    # save last line
    comment = SSAEvent(event.start, event.end, curr_line.raw, type="Comment")
    subs.append(comment)
    event.text = event.text.strip()
    subs.append(event)

    return subs
