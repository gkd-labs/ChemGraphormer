"""
Combine all labeled shards (labeled_shard_*.pt from add_labels_sharded.py) into ONE
ordinary torch file (torch.save / torch.load), WITHOUT ever holding the dataset in RAM.

Resumable: re-run the same command if interrupted.

The final file contains:  {"format", "data": {x, edge_index_T, edge_attr, lap_pos, y},
                           "slices": {x, edge_index_T, edge_attr, lap_pos, y}}
(edge_index is stored transposed, as [num_edges, 2], so shards can be appended.)

Read it with CombinedGraphs (below), which gives random access to any graph:
    from combine_shards import CombinedGraphs
    ds = CombinedGraphs("final.pt")      # memory-mapped, uses almost no RAM
    len(ds); ds[123]                     # a PyG Data object
    DataLoader(ds, batch_size=256, shuffle=True)   # full global shuffling works

Usage:
    python combine_shards.py --labeled-dir labels --output final.pt
"""
import glob
import json
import os
import shutil

import numpy as np
import torch
from tqdm import tqdm

KEYS = ["x", "edge_index_T", "edge_attr", "lap_pos", "y"]
# which per-graph count each key's slices follow
SLICE_BY = {"x": "nodes", "lap_pos": "nodes", "edge_index_T": "edges", "edge_attr": "edges", "y": "graphs"}
FORMAT = "collated-graphs-v1"


def _flatten(graphs):
    """Concatenate one shard's graphs into per-key 2D arrays + per-graph node/edge counts."""
    n_nodes = np.array([g.x.shape[0] for g in graphs], dtype=np.int64)
    n_edges = np.array([g.edge_index.shape[1] for g in graphs], dtype=np.int64)
    arrays = {
        "x": torch.cat([g.x for g in graphs], 0),
        "edge_index_T": torch.cat([g.edge_index.t() for g in graphs], 0),
        "edge_attr": torch.cat([g.edge_attr for g in graphs], 0),
        "lap_pos": torch.cat([g.lap_pos for g in graphs], 0),
        "y": torch.cat([g.y.reshape(-1, 1) for g in graphs], 0),
    }
    return {k: v.contiguous().numpy() for k, v in arrays.items()}, n_nodes, n_edges


def _read_progress(work):
    """Return (config, list of progress entries). Ignores a half-written last line."""
    cfg_path, prog_path = os.path.join(work, "config.json"), os.path.join(work, "progress.jsonl")
    if not os.path.exists(cfg_path):
        return None, []
    with open(cfg_path) as f:
        cfg = json.load(f)
    entries = []
    if os.path.exists(prog_path):
        with open(prog_path, "rb") as f:
            for line in f.read().split(b"\n"):
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    return cfg, entries


def combine_labeled_shards(labeled_dir, output_path, skip_disk_check=False):
    if os.path.exists(output_path):
        raise FileExistsError(f"{output_path} already exists; refusing to overwrite it.")

    work = output_path + ".work"
    meta_dir = os.path.join(work, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(labeled_dir, "labeled_shard_*.pt")))
    cfg, entries = _read_progress(work)
    done = [e["name"] for e in entries]

    # --- resume bookkeeping: cut work files back to the last fully recorded state ---
    if entries:
        sizes = entries[-1]["sizes"]
        for k in KEYS:
            p = os.path.join(work, f"{k}.bin")
            if os.path.exists(p) and os.path.getsize(p) > sizes[k]:
                with open(p, "r+b") as f:
                    f.truncate(sizes[k])
        for p in paths:                                   # copied earlier but not yet deleted
            if os.path.basename(p) in set(done):
                os.remove(p)
        paths = [p for p in paths if os.path.basename(p) not in set(done)]
        print(f"Resuming: {len(done)} shards already converted, {len(paths)} left")
    elif not paths:
        raise FileNotFoundError(f"No labeled_shard_*.pt files in {labeled_dir}")
    else:
        for k in KEYS:                                    # fresh start: make sure bins are empty
            open(os.path.join(work, f"{k}.bin"), "wb").close()

    # --- disk check: final file is written while the work files still exist ---
    if not skip_disk_check:
        need = sum(os.path.getsize(p) for p in paths) + sum(
            os.path.getsize(os.path.join(work, f"{k}.bin")) for k in KEYS)
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(output_path))).free
        gb = 1024 ** 3
        print(f"Disk: ~{need / gb:.1f} GB of data, {free / gb:.1f} GB free "
              f"(about 1x the data size must be free for the final file)")
        if free < need * 1.05:
            raise OSError("Probably not enough free disk space for the final file. "
                          "Nothing was deleted. Free some space or use --skip-disk-check.")

    # --- stream shards into the work files, deleting each one after it is safely recorded ---
    handles = {k: open(os.path.join(work, f"{k}.bin"), "ab") for k in KEYS}
    prog = open(os.path.join(work, "progress.jsonl"), "ab")
    try:
        for p in tqdm(paths, desc="Moving shards into work files"):
            graphs = torch.load(p, weights_only=False)
            if len(graphs) == 0:
                os.remove(p)
                continue
            if cfg is None:
                g0 = graphs[0]
                if set(g0.to_dict().keys()) != {"x", "edge_index", "edge_attr", "lap_pos", "y"}:
                    raise ValueError(f"Unexpected Data keys: {list(g0.to_dict().keys())}")
                arrays, _, _ = _flatten(graphs[:1])
                cfg = {k: {"dtype": str(arrays[k].dtype), "width": int(arrays[k].shape[1])} for k in KEYS}
                with open(os.path.join(work, "config.json"), "w") as f:
                    json.dump(cfg, f)

            arrays, n_nodes, n_edges = _flatten(graphs)
            for k in KEYS:
                if str(arrays[k].dtype) != cfg[k]["dtype"] or arrays[k].shape[1] != cfg[k]["width"]:
                    raise ValueError(f"{p}: '{k}' has dtype/width {arrays[k].dtype}/{arrays[k].shape[1]}, "
                                     f"expected {cfg[k]['dtype']}/{cfg[k]['width']}")
                arrays[k].tofile(handles[k])
                handles[k].flush()
                os.fsync(handles[k].fileno())

            name = os.path.basename(p)
            np.savez(os.path.join(meta_dir, name + ".npz"), n_nodes=n_nodes, n_edges=n_edges)
            sizes = {k: handles[k].tell() for k in KEYS}
            prog.write((json.dumps({"name": name, "n_graphs": len(graphs), "sizes": sizes}) + "\n").encode())
            prog.flush()
            os.fsync(prog.fileno())
            entries.append({"name": name, "n_graphs": len(graphs), "sizes": sizes})

            del graphs, arrays
            os.remove(p)              # recorded durably in the work files: free this shard's disk space
    finally:
        for h in handles.values():
            h.close()
        prog.close()

    if not entries:
        raise RuntimeError("No graphs were converted.")

    # --- build offset arrays ---
    n_nodes = np.concatenate([np.load(os.path.join(meta_dir, e["name"] + ".npz"))["n_nodes"] for e in entries])
    n_edges = np.concatenate([np.load(os.path.join(meta_dir, e["name"] + ".npz"))["n_edges"] for e in entries])
    n_graphs = len(n_nodes)
    counts = {"nodes": n_nodes, "edges": n_edges, "graphs": np.ones(n_graphs, dtype=np.int64)}
    slices = {}
    for k in KEYS:
        c = counts[SLICE_BY[k]]
        slices[k] = torch.from_numpy(np.concatenate([[0], np.cumsum(c)]).astype(np.int64))

    # --- memory-map the work files (no RAM) and let ONE torch.save stream them out ---
    data = {}
    for k in KEYS:
        rows = int(slices[k][-1])
        width, dtype = cfg[k]["width"], np.dtype(cfg[k]["dtype"])
        path = os.path.join(work, f"{k}.bin")
        expected = rows * width * dtype.itemsize
        if os.path.getsize(path) != expected:
            raise RuntimeError(f"{k}.bin has {os.path.getsize(path)} bytes, expected {expected}.")
        if rows == 0:
            data[k] = torch.empty((0, width), dtype=getattr(torch, cfg[k]["dtype"].replace("torch.", "")))
        else:
            data[k] = torch.from_numpy(np.memmap(path, dtype=dtype, mode="r+", shape=(rows, width)))

    print(f"Writing final file: {n_graphs} graphs, {int(n_nodes.sum())} nodes, {int(n_edges.sum())} edges")
    tmp = output_path + ".tmp"
    torch.save({"format": FORMAT, "data": data, "slices": slices}, tmp)
    del data
    os.replace(tmp, output_path)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Final file was not written correctly; work files were kept.")
    shutil.rmtree(work)
    print(f"Saved {n_graphs} labeled graphs to {output_path} ({os.path.getsize(output_path) / 1024**3:.2f} GB); "
          f"labeled shards and work files deleted.")


class CombinedGraphs:
    """Random-access, memory-mapped view of the file made by combine_labeled_shards."""

    def __init__(self, path, mmap=True):
        blob = torch.load(path, mmap=mmap, weights_only=True)
        if blob.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file.")
        self.data = blob["data"]
        self.sl = {k: v.numpy() for k, v in blob["slices"].items()}

    def __len__(self):
        return len(self.sl["y"]) - 1

    def __getitem__(self, i):
        from torch_geometric.data import Data
        if i < 0:
            i += len(self)
        s, d = self.sl, self.data
        a, b = s["x"][i], s["x"][i + 1]
        ea, eb = s["edge_index_T"][i], s["edge_index_T"][i + 1]
        return Data(
            x=d["x"][a:b].clone(),
            edge_index=d["edge_index_T"][ea:eb].t().contiguous(),
            edge_attr=d["edge_attr"][ea:eb].clone(),
            lap_pos=d["lap_pos"][a:b].clone(),
            y=d["y"][i:i + 1].reshape(-1).clone(),
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled-dir", required=True, help="Folder with labeled_shard_*.pt")
    ap.add_argument("--output", required=True, help="Path of the single combined .pt file")
    ap.add_argument("--skip-disk-check", action="store_true")
    args = ap.parse_args()

    combine_labeled_shards(args.labeled_dir, args.output, skip_disk_check=args.skip_disk_check)