"""Inspect safetensors model metadata: name, shape, dtype for each tensor.

Prints the full per-tensor table followed by a merged summary table, where
keys following the same digit-indexed pattern (e.g. layers.0/1/2) are
summarized the same way transformers' loading_report.update_key_name merges
them (see _merged_key). Only keys sharing the exact same shape and dtype are
merged together.

Only each file's JSON header is read; the tensor data (often tens of GB) is
never touched, so this stays fast even for very large models.
"""

import argparse
import json
import re
import struct
from pathlib import Path

# Same "dot-delimited integer" pattern the transformers loader uses to build
# merge buckets, so per-key counts line up with update_key_name's grouping.
_DIGIT_RX = re.compile(r"(?<=\.)(\d+)(?=\.|$)")

# Bytes per element for every dtype the safetensors format supports.
_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "F8_E4M3": 8
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
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = tuple(meta["shape"])
        dtype = meta["dtype"]
        elems = 1
        for dim in shape:
            elems *= dim
        yield name, shape, dtype, elems * _DTYPE_BYTES[dtype]


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


def main():
    parser = argparse.ArgumentParser(description="Inspect safetensors model metadata")
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()

    index_path = args.model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            shard_names = sorted(set(json.load(f)["weight_map"].values()))
        files = [args.model_dir / s for s in shard_names]
    else:
        files = sorted(args.model_dir.glob("*.safetensors"))

    if not files:
        print("No safetensors files found.")
        return

    rows = []
    for f in files:
        rows.extend(parse_header(f))

    #name_w = max(len(r[0]) for r in rows)
    #shape_w = max(len(str(r[1])) for r in rows)
    #dtype_w = max(len(r[2]) for r in rows)
    #body = [(name, str(shape), dtype) for name, shape, dtype, _ in rows]
    #print_table(["Name", "Shape", "Dtype"], [name_w, shape_w, dtype_w], body)
    #print(f"\nTotal: {len(rows)} tensors")

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
