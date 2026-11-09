"""
Combine all labeled shards (labeled_shard_*.pt from add_labels_sharded.py) into a
single .pt file containing one list of PyG Data objects, in original CSV order.

After the final file has been written successfully, ALL the labeled shards that
were combined are DELETED.

NOTE: the whole dataset must fit in RAM at once. The script estimates the needed
memory from the shard sizes and stops before loading anything if it won't fit
(override with --force).

Usage:
    python combine_shards.py --labeled-dir labels --output final.pt
"""
import glob
import os

import torch
from tqdm import tqdm


def _mem_available_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def combine_labeled_shards(labeled_dir, output_path, force=False):
    if os.path.exists(output_path):
        raise FileExistsError(f"{output_path} already exists; refusing to overwrite it.")

    paths = sorted(glob.glob(os.path.join(labeled_dir, "labeled_shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No labeled_shard_*.pt files in {labeled_dir}")

    on_disk = sum(os.path.getsize(p) for p in paths)
    est = int(on_disk * 1.5)  # rough: in-memory size of the Data objects vs. file size
    avail = _mem_available_bytes()
    gb = 1024 ** 3
    print(f"Combining {len(paths)} shards: ~{on_disk / gb:.1f} GB on disk, "
          f"estimated ~{est / gb:.1f} GB RAM needed"
          + (f", {avail / gb:.1f} GB available" if avail else ""))
    if avail is not None and est > avail and not force:
        raise MemoryError(
            "The combined dataset probably does not fit in available RAM. "
            "Nothing was loaded or deleted. Re-run with --force to try anyway."
        )

    all_data = []
    for p in tqdm(paths, desc="Loading shards"):
        all_data.extend(torch.load(p, weights_only=False))
    n_graphs = len(all_data)

    tmp = output_path + ".tmp"
    torch.save(all_data, tmp)
    os.replace(tmp, output_path)
    del all_data

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Final file was not written correctly; shards were NOT deleted.")
    print(f"Saved {n_graphs} labeled graphs to {output_path}")

    # Final file is safely on disk: remove the shards that went into it.
    for p in tqdm(paths, desc="Deleting shards"):
        os.remove(p)
    print(f"Deleted {len(paths)} labeled shards from {labeled_dir}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled-dir", required=True, help="Folder with labeled_shard_*.pt")
    ap.add_argument("--output", required=True, help="Path of the single combined .pt file")
    ap.add_argument("--force", action="store_true",
                    help="Combine even if the estimated size exceeds available RAM.")
    args = ap.parse_args()

    combine_labeled_shards(args.labeled_dir, args.output, force=args.force)
