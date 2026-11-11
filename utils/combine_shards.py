"""
Combine all labeled shards (labeled_shard_*.pt from add_labels_sharded.py) into a
SINGLE file, streaming one shard at a time with constant memory.

Each labeled shard is copied byte-for-byte into the final file (no loading into
RAM), fsynced, and then that exact shard file is DELETED before the next one is
processed. A footer index is written at the end so any shard can be read back
individually.

The final file is NOT a plain torch.load()-able list (a normal .pt cannot be
appended to). Read it with the ShardedPT class below:

    from combine_shards import ShardedPT
    ds = ShardedPT("final.pt")
    len(ds)            # number of shards
    shard = ds[0]      # list of PyG Data objects (loads only that shard)
    for shard in ds:   # iterate shard by shard
        ...

Resumable: if interrupted, re-run the same command. Partially written trailing
data is detected and discarded, shards already inside the file are not copied
again, and any leftover duplicate source shard is deleted.

Usage:
    python combine_shards.py --labeled-dir labels --output final.pt
"""
import glob
import io
import json
import os
import shutil
import struct

import torch
from tqdm import tqdm

REC_MAGIC = b"SHRD"
END_MAGIC = b"SHRDEND1"
REC_HDR = struct.Struct("<4sHQ")   # record magic, name length, data length
TAIL = struct.Struct("<Q8s")       # index json length, end magic


def _scan_records(f, file_size):
    """Walk the complete records already in the file. Returns (index, end_of_last_good_record)."""
    index, pos = [], 0
    while pos + REC_HDR.size <= file_size:
        f.seek(pos)
        magic, name_len, data_len = REC_HDR.unpack(f.read(REC_HDR.size))
        if magic != REC_MAGIC:
            break
        end = pos + REC_HDR.size + name_len + data_len
        if end > file_size:
            break  # partially written record
        name = f.read(name_len).decode()
        index.append({"name": name, "offset": pos + REC_HDR.size + name_len, "length": data_len})
        pos = end
    return index, pos


def combine_labeled_shards(labeled_dir, output_path):
    if os.path.exists(output_path):
        raise FileExistsError(f"{output_path} already exists; refusing to overwrite it.")

    tmp = output_path + ".tmp"
    paths = sorted(glob.glob(os.path.join(labeled_dir, "labeled_shard_*.pt")))
    resuming = os.path.exists(tmp)
    if not paths and not resuming:
        raise FileNotFoundError(f"No labeled_shard_*.pt files in {labeled_dir}")

    with open(tmp, "r+b" if resuming else "w+b") as f:
        index = []
        if resuming:
            size = os.path.getsize(tmp)
            index, good_end = _scan_records(f, size)
            if good_end < size:
                f.truncate(good_end)   # drop a half-written record / stale footer
            f.seek(good_end)
            done = {e["name"] for e in index}
            for p in paths:            # crashed after copying but before deleting
                if os.path.basename(p) in done:
                    os.remove(p)
            paths = [p for p in paths if os.path.basename(p) not in done]
            print(f"Resuming: {len(index)} shards already in {tmp}, {len(paths)} left")

        for p in tqdm(paths, desc="Moving shards into final file"):
            name = os.path.basename(p)
            name_b = name.encode()
            size = os.path.getsize(p)

            f.write(REC_HDR.pack(REC_MAGIC, len(name_b), size))
            f.write(name_b)
            data_off = f.tell()
            with open(p, "rb") as src:
                shutil.copyfileobj(src, f, 16 * 1024 * 1024)
            if f.tell() - data_off != size:
                raise RuntimeError(f"Short copy of {p}; shard NOT deleted.")

            f.flush()
            os.fsync(f.fileno())       # bytes are on disk before we delete the source
            index.append({"name": name, "offset": data_off, "length": size})
            os.remove(p)               # free this shard's disk space right away

        idx = json.dumps(index).encode()
        f.write(idx)
        f.write(TAIL.pack(len(idx), END_MAGIC))
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, output_path)
    print(f"Wrote {len(index)} shards to {output_path} ({os.path.getsize(output_path) / 1024**3:.2f} GB); "
          f"all source shards deleted.")


class ShardedPT:
    """Random access to the shards inside a file made by combine_labeled_shards."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            f.seek(-TAIL.size, os.SEEK_END)
            n, magic = TAIL.unpack(f.read(TAIL.size))
            if magic != END_MAGIC:
                raise ValueError(f"{path} is not a complete combined shard file (missing footer).")
            f.seek(-(TAIL.size + n), os.SEEK_END)
            self.index = json.loads(f.read(n))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e = self.index[i]
        with open(self.path, "rb") as f:
            f.seek(e["offset"])
            buf = f.read(e["length"])
        return torch.load(io.BytesIO(buf), weights_only=False)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled-dir", required=True, help="Folder with labeled_shard_*.pt")
    ap.add_argument("--output", required=True, help="Path of the single combined file")
    args = ap.parse_args()

    combine_labeled_shards(args.labeled_dir, args.output)
