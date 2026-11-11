"""
Sharded, resumable runner for graph_generator.py

Processes the dataset in shards (default 100k molecules), saves each shard to
its own .pt file as soon as it finishes, and keeps memory flat. If the run is
interrupted, re-running the same command skips shards that already exist.

Each shard file contains four lists plus "indices":
the ROW NUMBERS (in the input CSV) of the molecules that were kept. Invalid
SMILES are skipped, and so these same "indices" will be used to line graphs up with their labels.

Usage:
    python utils/run_sharded.py --input dataset.csv --out-dir dataset --shard-size 100000
"""
import os

# Silence the torchdata/dgl deprecation warning (also in spawned workers, since they re-import this module)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchdata.*")
warnings.filterwarnings("ignore", message=r"(?s).*datapipes.*deprecated.*")

import argparse
import logging
import multiprocessing as mp

import torch
from tqdm import tqdm

from graph_generator import _init_worker, _process_one, get_num_workers

logger = logging.getLogger("run_sharded")


def run_sharded(smiles_list, out_dir, k=51, n_jobs=None, shard_size=100_000, desc_total=None):
    os.makedirs(out_dir, exist_ok=True)
    n = len(smiles_list)
    n_jobs = get_num_workers() if n_jobs is None else max(1, int(n_jobs))
    n_shards = (n + shard_size - 1) // shard_size
    logger.info(f"{n} molecules -> {n_shards} shards of up to {shard_size}, {n_jobs} worker(s)")

    ctx = mp.get_context("spawn")
    chunksize = 64
    with ctx.Pool(processes=n_jobs, initializer=_init_worker, initargs=(k,)) as pool:
        for s in range(n_shards):
            path = os.path.join(out_dir, f"shard_{s:05d}.pt")
            if os.path.exists(path):
                logger.info(f"shard {s + 1}/{n_shards} exists, skipping")
                continue

            lo, hi = s * shard_size, min((s + 1) * shard_size, n)
            chunk = smiles_list[lo:hi]

            nodes, edges, pairs, laps, indices = [], [], [], [], []
            it = pool.imap(_process_one, chunk, chunksize=chunksize)  # ordered
            for j, r in enumerate(tqdm(it, total=len(chunk), desc=f"shard {s + 1}/{n_shards}", leave=False)):
                if r is None:
                    continue
                node_np, edge_np, p, lap_np = r
                nodes.append(torch.from_numpy(node_np))
                edges.append(torch.from_numpy(edge_np))
                pairs.append(p)
                laps.append(torch.from_numpy(lap_np))
                indices.append(lo + j)

            tmp = path + ".tmp"
            torch.save(
                {
                    "node_features": nodes,
                    "edge_features": edges,
                    "edge_pairs": pairs,
                    "laplacian_pe": laps,
                    "indices": indices,
                },
                tmp,
            )
            os.replace(tmp, path)  # atomic: a half-written shard is never mistaken for a finished one
            logger.info(f"shard {s + 1}/{n_shards} saved: kept {len(indices)}/{len(chunk)} -> {path}")
            del nodes, edges, pairs, laps, indices


if __name__ == "__main__":
    import pandas as pd

    logging.basicConfig(level=logging.INFO)

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--k", type=int, default=51)
    p.add_argument("--n-jobs", type=int, default=None, help="default: all available cores")
    p.add_argument("--shard-size", type=int, default=100_000)
    p.add_argument("--out-dir", default="shards")
    args = p.parse_args()

    smiles_list = pd.read_csv(args.input)[args.smiles_col].tolist()
    run_sharded(smiles_list, args.out_dir, k=args.k, n_jobs=args.n_jobs, shard_size=args.shard_size)
