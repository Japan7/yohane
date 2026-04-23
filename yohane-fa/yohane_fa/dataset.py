import re
import unicodedata
from typing import cast

from datasets import load_dataset as _load_dataset


def load_dataset(
    path: str,
    *,
    split: str,
    max_duration: int,
    load_from_cache_file: bool,
):
    dataset = _load_dataset(path, split=split)
    dataset = dataset.filter(
        lambda x: x.metadata.duration_seconds <= max_duration,
        input_columns=["audio"],
        new_fingerprint="yohane_fa_filtered",
        load_from_cache_file=load_from_cache_file,
    )
    dataset = dataset.map(
        normalize_morae,
        input_columns=["morae"],
        writer_batch_size=500,
        new_fingerprint="yohane_fa_normalized",
    )
    return dataset


def normalize_morae(morae):
    for mora in morae:
        value = cast(str, mora["value"])
        value = value.casefold()
        value = value.replace("’", "'")
        value = unicodedata.normalize("NFKD", value)
        value = value.encode("ascii", "ignore").decode("ascii")
        value = re.sub(r"[^a-z']", " ", value)
        value = re.sub(r" +", " ", value)
        mora["value"] = value
    return {"morae": morae}
