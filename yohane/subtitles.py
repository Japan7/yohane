from pysubs2 import SSAEvent, SSAFile
from torch import Tensor
from torchaudio.functional import TokenSpan

from yohane.audio_processing import bundle
from yohane.text_processing import Text


def make_ass(
    lyrics_txt: Text,
    waveform: Tensor,
    emission: Tensor,
    token_spans: list[list[TokenSpan]],
):
    # audio processing parameters
    num_frames = emission.size(1)
    ratio = waveform.size(1) / num_frames
    sample_rate = bundle.sample_rate

    # init subs and event
    subs = SSAFile()
    event = SSAEvent(-1)

    # init iterators
    lines_iter = iter(lyrics_txt.lines)
    curr_line = next(lines_iter)
    words_iter = iter(curr_line.words)
    curr_word = next(words_iter)
    syllables_iter = iter(curr_word.syllables)
    curr_syllable = next(syllables_iter)

    # iterate over all aligned tokens
    for i, spans in enumerate(token_spans):
        # start and end time of token
        x0 = ratio * spans[0].start
        x1 = ratio * spans[-1].end
        t_start = x0 / sample_rate
        t_end = x1 / sample_rate

        # set event start time if is a new one
        if event.start == -1:
            event.start = int(t_start * 1000)
        # set event end time
        event.end = int(t_end * 1000)

        # k tag logic:
        # snap the token end time to the next token start time
        # if it is not the last in the line
        if i < len(token_spans) - 1:
            next_x0 = ratio * token_spans[i + 1][0].start
            next_t_start = next_x0 / sample_rate
            k_end = next_t_start
        else:
            k_end = t_end
        k_duration = round((k_end - t_start) * 100)

        event.text += rf"{{\k{k_duration}}}{curr_syllable}"

        # if that was the last token, don't enter the iter logic
        if i == len(token_spans) - 1:
            break

        try:
            # advance to next syllable
            curr_syllable = next(syllables_iter)
        except StopIteration:
            try:
                # no more syllables in word:
                # add a space, advance to next word

                event.text += " "

                # iterators logic
                curr_word = next(words_iter)
                syllables_iter = iter(curr_word.syllables)
                curr_syllable = next(syllables_iter)
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

                    # iterators logic
                    curr_line = next(lines_iter)
                    words_iter = iter(curr_line.words)
                    curr_word = next(words_iter)
                    syllables_iter = iter(curr_word.syllables)
                    curr_syllable = next(syllables_iter)
                except StopIteration as e:
                    raise RuntimeError("should not happen") from e

    # save last line
    comment = SSAEvent(event.start, event.end, curr_line.raw, type="Comment")
    subs.append(comment)
    event.text = event.text.strip()
    subs.append(event)

    return subs
