"""Inspect safetensors model metadata: name, shape, dtype for each tensor.

Prints the full per-tensor table followed by a merged summary table, where
keys following the same digit-indexed pattern (e.g. layers.0/1/2) are
summarized the same way transformers' loading_report.update_key_name merges
them (see _merged_key). Only keys sharing the exact same shape and dtype are
merged together.

Only each file's JSON header is read; the tensor data (often tens of GB) is
never touched, so this stays fast even for very large models.

The model may be a local directory of *.safetensors files (with or without a
model.safetensors.index.json), or a Hugging Face model stub such as
"meta-llama/Llama-3.2-1B-Instruct". Hub models already downloaded to the
local HF cache are inspected from the cached files; otherwise only the
index and each shard's JSON header are fetched over HTTP, so the weights
are never downloaded (requires huggingface_hub).
"""

import argparse
import json
import re
import struct
import urllib.request
from pathlib import Path

# Same "dot-delimited integer" pattern the transformers loader uses to build
# merge buckets, so per-key counts line up with update_key_name's grouping.
_DIGIT_RX = re.compile(r"(?<=\.)(\d+)(?=\.|$)")

# Bytes per element for every dtype the safetensors format supports.
_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U8": 1, "F8_E4M3": 8, "F8_E8M0": 8,
}


def _pattern_of(key: str) -> str:
    return _DIGIT_RX.sub("*", key)


def _fmt_indices(values, cutoff=10):
    # Mirrors transformers.utils.loading_report._fmt_indices.
    if len(values) == 1:
        return str(values[0])
    values = sorted(values)
    if len(values) > cutoff:
        return f"{values[0]}...{values[-1]}"
    return ", ".join(map(str, values))


def _merged_key(pattern, index_sets):
    # Mirrors the key-merging of transformers.utils.loading_report
    # .update_key_name in non-mapping mode.
    parts = pattern.split("*")
    final = parts[0]
    for index_set, part in zip(index_sets, parts[1:]):
        vals = sorted(index_set)
        if vals:
            insert = _fmt_indices(vals)
            final += (f"{{{insert}}}" if len(vals) > 1 else insert)
        else:
            final += "*"
        final += part
    return final


def parse_header(path: Path):
    """Yield (name, shape, dtype, nbytes) for each tensor in the file.

    Reads only the 8-byte length prefix and the JSON header, never the
    tensor bytes themselves.
    """
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    yield from _parse_header(header)


def _parse_header(header):
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = tuple(meta["shape"])
        dtype = meta["dtype"]
        elems = 1
        for dim in shape:
            elems *= dim
        yield name, shape, dtype, elems * _DTYPE_BYTES[dtype]


def _resolve_hub_files(model_id: str):
    """List safetensors shard filenames of a Hub model without downloading.

    Returns (repo_id, revision, [shard names]). Raises ValueError if the
    model does not exist on the Hub.
    """
    from huggingface_hub import hf_hub_url, list_repo_files
    from huggingface_hub.utils import (
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    model_id, revision = _split_model_id(model_id)

    try:
        files = list_repo_files(repo_id=model_id, revision=revision)
    except (RepositoryNotFoundError, RevisionNotFoundError) as e:
        raise ValueError(f"{model_id}: model not found on the Hugging Face Hub") from e

    shards = [f for f in files if f.endswith(".safetensors")]
    if "model.safetensors.index.json" in files:
        url = hf_hub_url(repo_id=model_id, revision=revision,
                         filename="model.safetensors.index.json")
        with urllib.request.urlopen(url) as r:
            weight_map = json.load(r)["weight_map"]
        shard_names = sorted(set(weight_map.values()))
    elif shards:
        shard_names = shards
    else:
        raise ValueError(f"{model_id}: no safetensors files on the Hub")

    return model_id, revision, shard_names


def fetch_hub_headers(model_id: str, revision, shard_names):
    """Yield (name, shape, dtype, nbytes) per tensor across Hub shards.

    Only each file's index and JSON header are fetched over HTTP; the tensor
    data is never requested.
    """
    from huggingface_hub import hf_hub_url

    for shard in shard_names:
        url = hf_hub_url(repo_id=model_id, revision=revision,
                         filename=shard)
        with urllib.request.urlopen(url) as r:
            (header_len,) = struct.unpack("<Q", r.read(8))
            header = json.loads(r.read(header_len))
        yield from _parse_header(header)


def _split_model_id(model_id: str):
    model_id = model_id.rstrip("/")
    if "@" in model_id:
        model_id, revision = model_id.split("@", 1)
    else:
        revision = None
    return model_id, revision


def _cached_snapshot_dir(model_id: str, revision):
    """Return the snapshot directory if the model is in the local HF cache.

    Uses huggingface_hub.try_to_load_from_cache to resolve model.safetensors
    (or the index for sharded models) without any network access, then
    returns the snapshot directory containing it. Returns None when the
    model (or a complete snapshot) is not cached.
    """
    from huggingface_hub import try_to_load_from_cache

    for filename in ("model.safetensors", "model.safetensors.index.json"):
        path = try_to_load_from_cache(model_id, filename, revision=revision)
        if isinstance(path, str):
            snapshot_dir = Path(path).parent
            if _rows_from_dir(snapshot_dir) and list(snapshot_dir.glob("*.safetensors")):
                return snapshot_dir
            break
    return None


def merged_rows(rows):
    """Group rows by (pattern, shape, dtype), then merge each group's keys.

    A merged key is only meaningful when every key in the group collapses to
    it, so grouping by structural pattern as well guarantees each group
    collapses to exactly one merged key and per-key counts stay accurate.
    """
    groups = {}
    for name, shape, dtype, nbytes in rows:
        groups.setdefault((_pattern_of(name), shape, dtype), []).append((name, nbytes))
    merged = []
    for (_pattern, shape, dtype), entries in groups.items():
        # All names in a group share the same pattern, hence the same count
        # of digit groups.
        index_sets = [set() for _ in _DIGIT_RX.findall(entries[0][0])]
        for name, _ in entries:
            for index_set, digit in zip(index_sets, _DIGIT_RX.findall(name)):
                index_set.add(int(digit))
        merged.append((
            _merged_key(_pattern, index_sets),
            list(shape),
            dtype,
            len(entries),
            sum(nbytes for _, nbytes in entries),
        ))
    return sorted(merged, key=lambda r: r[0])


def print_table(headers, widths, body):
    header = "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(header)
    print("-" * len(header))
    for line in body:
        print("  ".join(f"{v:<{w}}" for v, w in zip(line, widths)))


def _rows_from_dir(model_dir: Path):
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            shard_names = sorted(set(json.load(f)["weight_map"].values()))
        files = [model_dir / s for s in shard_names]
    else:
        files = sorted(model_dir.glob("*.safetensors"))

    for f in files:
        yield from parse_header(f)


def _rows_from_hub(model_id: str):
    """Read tensor rows for a Hub model, preferring the local HF cache.

    If the model is already downloaded to the local Hugging Face cache, the
    cached snapshot files are used directly. Otherwise the model is
    inspected over HTTP, fetching only the index and each shard's JSON
    header so the weights are never downloaded.
    """
    model_id, revision = _split_model_id(model_id)

    snapshot_dir = _cached_snapshot_dir(model_id, revision)
    if snapshot_dir is not None:
        return _rows_from_dir(snapshot_dir)

    model_id, revision, shard_names = _resolve_hub_files(model_id)
    return fetch_hub_headers(model_id, revision, shard_names)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect safetensors model metadata",
        epilog="MODEL is a local directory or a Hugging Face model stub "
               "such as meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument("model", metavar="MODEL",
                        help="model directory or Hugging Face model stub")
    args = parser.parse_args()

    model_path = Path(args.model)
    if model_path.is_dir():
        rows = list(_rows_from_dir(model_path))
    else:
        rows = list(_rows_from_hub(args.model))

    #name_w = max(len(r[0]) for r in rows)
    #shape_w = max(len(str(r[1])) for r in rows)
    #dtype_w = max(len(r[2]) for r in rows)
    #body = [(name, str(shape), dtype) for name, shape, dtype, _ in rows]
    #print_table(["Name", "Shape", "Dtype"], [name_w, shape_w, dtype_w], body)
    #print(f"\nTotal: {len(rows)} tensors")

    if not rows:
        print("No safetensors files found.")
        return

    print()
    merged = merged_rows(rows)
    total_bytes = sum(r[3] for r in rows)
    name_w = max(len(r[0]) for r in merged)
    shape_w = max(len(str(r[1])) for r in merged)
    dtype_w = max(len(r[2]) for r in merged)
    keys_w = max(len(str(r[3])) for r in merged)
    pcts = [f"{100.0 * nbytes / total_bytes:.3f}%" for _, _, _, _, nbytes in merged]
    pct_w = max(len(p) for p in pcts)
    merged_body = [
        (name, str(shape), dtype, nkeys, pct)
        for (name, shape, dtype, nkeys, _), pct in zip(merged, pcts)
    ]
    print_table(["Key", "Shape", "Dtype", "Keys", "Perc total"],
                [name_w, shape_w, dtype_w, keys_w, pct_w], merged_body)


if __name__ == "__main__":
    main()
