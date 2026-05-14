import re
import unicodedata
from typing import TYPE_CHECKING, cast

from datasets import load_dataset as _load_dataset

if TYPE_CHECKING:
    from yohane_fa.lightning import DatasetMora


def load_dataset(
    path: str,
    *,
    split: str,
    max_duration: int,
    load_from_cache_file: bool,
):
    dataset = _load_dataset(path, split=split)
    dataset = dataset.map(
        normalize_morae,
        input_columns=["morae"],
        new_fingerprint="yohane_fa_normalize",
        load_from_cache_file=load_from_cache_file,
    )
    dataset = dataset.map(
        lambda morae: {
            "morae": sorted(
                [mora for mora in morae if mora["value"].strip()],
                key=lambda mora: mora["start"],
            )
        },
        input_columns=["morae"],
        new_fingerprint="yohane_fa_sort",
        load_from_cache_file=load_from_cache_file,
    )
    dataset = dataset.filter(
        lambda morae: len(morae) > 0,
        input_columns=["morae"],
        new_fingerprint="yohane_fa_nonempty",
        load_from_cache_file=load_from_cache_file,
    )
    dataset = dataset.filter(
        lambda x: x.metadata.duration_seconds <= max_duration,
        input_columns=["audio"],
        new_fingerprint=f"yohane_fa_filter_{max_duration}s",
        load_from_cache_file=load_from_cache_file,
    )
    return dataset


def normalize_morae(morae: list["DatasetMora"]):
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
