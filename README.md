# ChemGraphormer

**A Bond-Aware Sparse Graph Attention Transformer for Molecular Property Prediction**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Overview

ChemGraphormer is a chemically inductive sparse graph attention Transformer that learns molecular representations exclusively from **2D molecular graphs** — no 3D coordinates, bond lengths, or conformers required. It introduces sparse bond-restricted dot-product attention computed only over directly bonded atom pairs via `scatter_softmax` / `scatter_add`, with triple simultaneous bond conditioning of the attention logit, value content, and message amplitude within a single unified attention operation.

**QM9 ZPVE results (standard benchmark split adapted from DimeNet— 110k/10k/10,831):**

| Model | ZPVE MAE on 10831 (meV) | 3D? |
|---|---|---|
| **ChemGraphormer — Mean Pool + Gate b=0** | **8.75** | **No** |
| **ChemGraphormer — Mean Pool + Gate b=1** | **8.37** | **No** |
| DimeNet++ | 1.210 | Yes |
| PaiNN | 1.280 | Yes |
| SchNet | 1.700 | Yes |

**OGB PCQM4Mv2:** validation MAE of **0.1013 eV** (rank 34/42) on a single NVIDIA L4 GPU (22.5 GB).

---

## 1. Clone Repository

**PLEASE NOTE: All experiments were conducted using Python 3.12**

```bash
git clone https://github.com/gkd-labs/ChemGraphormer.git
cd ChemGraphormer
```

---

## 2. Install Requirements


```bash
pip install -r requirements.txt
```

---

## 3. Imports Libraries

```python
import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Graph generators
from ablate.graph_generator_ablate import run_graph_generator_ablate
from utils.graph_generator_ogb import run_graph_generator_ogb

# QM9 ablation training pipelines
from ablate.chemgraphormer_ablate_edge_training_pipeline import run_edge_ablation
from ablate.chemgraphormer_ablate_gate_two_training_pipeline import run_gate_init_two_ablation
from ablate.chemgraphormer_ablate_gate_zero_training_pipeline import run_gate_init_zero_ablation
from ablate.chemgraphormer_ablate_gate_one_training_pipeline import run_gate_init_one_ablation
from ablate.chemgraphormer_ablate_rpe_edge_training_pipeline import run_rpe_edge_ablation
from ablate.chemgraphormer_ablate_rpe_training_pipeline import run_rpe_ablation
from ablate.chemgraphormer_ablate_no_edge_msg_and_gate_training_pipeline import run_no_edge_msg_and_gate_ablation
from ablate.chemgraphormer_ablate_no_sinusoidal_rpe_training_pipeline import run_no_sinusoidal_rpe_ablation
from ablate.chemgraphormer_ablate_use_mean_pooling_gate_one_training_pipeline import run_use_mean_pooling_gate_one_ablation
from ablate.chemgraphormer_ablate_use_mean_pooling_gate_zero_training_pipeline import run_use_mean_pooling_gate_zero_ablation
from ablate.chemgraphormer_ablate_use_mean_pooling_gate_two_training_pipeline import run_use_mean_pooling_gate_two_ablation
from ablate.chemgraphormer_ablate_static_edge_flow_training_pipeline import run_static_edge_flow_ablation

# QM9 prediction and evaluation
from ablate.predict_zpve_values_attnpl import make_prediction_attnpl
from ablate.predict_zpve_values_meanpl import make_prediction_meanpl
from ablate.compute_convergence_efficiency import compute_convergence_training_efficiency

# OGB training and prediction
from utils.train_chemgraphormer_meanp_reactive import train_with_meanp_gate0_reactive
from utils.make_gap_value_prediction import run_prediction_meanp
```

---

## 4. QM9 Dataset Preparation

Download QM9 from HuggingFace and preprocess:

```python
python load_qm9_130831.py
```

Read and split into standard benchmark splits (adopted from DimeNet):

```python
qm9_v = pd.read_csv("/content/qm9_130831.csv")

# train : val : test = 110,000 : 10,000 : 10,831
train_val, test = train_test_split(
    qm9_v[["smiles", "zero_point_energy"]], test_size=10831, random_state=42
)
train, val = train_test_split(train_val, test_size=10000, random_state=42)

# Extract SMILES and labels independently
# (smiles_df is a placeholder — repeat for train, val, and test)
smiles_df  = pd.DataFrame(smiles_df["smiles"])
labels     = pd.DataFrame(smiles_df["zero_point_energy"])
labels     = torch.tensor(labels.values, dtype=torch.float32)
```

---

## 5. Graph Data Computation

> **Note:** Input SMILES must be a Python DataFrame with a column named `"smiles"`.
> Graph computation for OGB follows the same steps below — replace `run_graph_generator_ablate` with `run_graph_generator_ogb`. The only difference is `k` in the Laplacian computation: **k = 9** for QM9 ablation, **k = 51** for OGB pretraining.

### Compute graphs

```python
if __name__ == "__main__":
    smiles_list = smiles_df["smiles"]
    node_features_tensor, edge_features_tensor, edges_indices, laplacian_pos = \
        run_graph_generator_ablate(smiles_list)

    all_graph_data = {
        'node_features': node_features_tensor,
        'edge_features': edge_features_tensor,
        'edge_indices':  edges_indices,
        'laplacian_pos': laplacian_pos,
    }

    save_dir = "/path/to/save"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "graph_data.pt")

    try:
        torch.save(all_graph_data, filepath)
    except Exception as e:
        print(f"Error saving molecular graph data: {e}")
```

### Load and assemble graph dataset

```python
graphs = torch.load('/path/graph_data.pt', weights_only=False)

node_features = graphs['node_features']
edge_features = graphs['edge_features']
edge_indices  = graphs['edge_indices']
laplacian_pos = graphs['laplacian_pos']

# Attach labels — required for training and validation sets
# Labels must be tensors
graph_dataset_list = []
for i in tqdm(range(len(node_features)), desc="Adding graph labels..."):
    data = Data(
        x          = node_features[i],
        edge_index = torch.tensor(edge_indices[i], dtype=torch.long).t().contiguous(),
        edge_attr  = edge_features[i],
        lap_pos    = laplacian_pos[i],
        y          = labels[i].unsqueeze(0)
    )
    graph_dataset_list.append(data)

torch.save(graph_dataset_list, "/path/graphs.pt")
```

---

## 6. Ablation Training

The example below runs **Gate Init Zero** (Group A). To run any other condition, replace `run_gate_init_zero_ablation` with the corresponding import and update `ckpt_dir` and `epoch_log_path` accordingly.

| Condition | Function |
|---|---|
| Gate b=0 (full model) | `run_gate_init_zero_ablation` |
| Gate b=1 (full model) | `run_gate_init_one_ablation` |
| Gate b=2 (full model) | `run_gate_init_two_ablation` |
| No edge bias | `run_edge_ablation` |
| No ΔRPE bias | `run_rpe_ablation` |
| No edge bias + no ΔRPE | `run_rpe_edge_ablation` |
| No sinusoidal RPE | `run_no_sinusoidal_rpe_ablation` |
| No edge msg + no gate | `run_no_edge_msg_and_gate_ablation` |
| Static edge flow (no gate) | `run_static_edge_flow_ablation` |
| Mean pool + gate b=0 | `run_use_mean_pooling_gate_zero_ablation` |
| Mean pool + gate b=1 | `run_use_mean_pooling_gate_one_ablation` |
| Mean pool + gate b=2 | `run_use_mean_pooling_gate_two_ablation` |

```python
args = argparse.Namespace(
    train_path               = "/path/train_graphs.pt",
    valid_path               = "/path/valid_graphs.pt",
    ckpt_dir                 = "/path/gate0/",
    epoch_log_path           = "/path/gate0/log_gate0.csv",
    num_classes              = 1,
    d_model                  = 512,
    d_ff                     = 1024,
    num_freqs                = 8,
    n_heads                  = 8,
    num_layers               = 6,
    epochs                   = 200,
    warmup_epochs            = 10,
    batch_size               = 128,
    lr                       = 8e-5,
    min_lr                   = 1e-6,
    weight_decay             = 5e-4,
    early_stopping_patience  = 40,
    dropout                  = 0.15,
    device                   = "cuda",
)

run_gate_init_zero_ablation(args)
```

---

## 7. Test Set Prediction and Convergence Efficiency

### Make predictions on test set

```python
gate0_model_dir  = "/path/gate0/chemgraphormer_gate_zero_ablation_best_model.pt"
test_dataset_path = "/path/test_graph_data.pt"

custom_config = {
    'num_classes': 1,
    'd_model':     512,
    'n_heads':     8,
    'num_layers':  6,
    'd_ff':        1024,
    'num_freqs':   8,
    'dropout':     0.0,
}

print(f"Test dataset size: {len(torch.load(test_dataset_path, weights_only=False))}")

gate0_ablate_pred, gate0_ablate_mae = [], []

preds, mae = make_prediction_attnpl(
    dataset_path    = test_dataset_path,
    checkpoint_path = gate0_model_dir,
    model_params    = custom_config,
    device          = "cuda",
    batch_size      = 128,
)

gate0_ablate_pred.append(preds)
gate0_ablate_mae.append(mae)

print("Predictions:", gate0_ablate_pred)
print("Test MAE:",    gate0_ablate_mae)
```

> **Important:** For mean pooling ablation variants (Group D), replace `make_prediction_attnpl` with `make_prediction_meanpl` — all other arguments remain identical.

### Compute convergence efficiency

```python
gate0_ablate_logs = pd.read_csv("/path/gate0/log_gate0.csv")

gate0_ablate_time = gate0_ablate_logs["Epoch Time (min)"].sum()
gate0_ablate_best_mae = gate0_ablate_logs["Val MAE"].min()
gate0_ablate_convergence_efficiency = compute_convergence_training_efficiency(
    gate0_ablate_time, gate0_ablate_best_mae
)

print("Convergence Efficiency:", gate0_ablate_convergence_efficiency)
```

---

## 8. OGB PCQM4Mv2 training

Note: Input SMILES must be a Python DataFrame with a column named "smiles".
Graph computation for OGB follows the same steps as Section 5, but replace replace `run_graph_generator_ablate` with `run_graph_generator_ogb`. The only difference is `k` in the Laplacian computation: `k = 9` for QM9 ablation and `k = 51` for OGB training.

### Run training

```python
args = argparse.Namespace(
    train_path      = "/path/train_graphs.pt",
    valid_path      = "/path/valid_graphs.pt",
    ckpt_dir        = "/path/ogb_meanp_reactive/",
    epoch_log_path  = "/path/ogb_meanp_reactive/train_log.csv",
    epochs          = 150,
    batch_size      = 1024,
    lr              = 1e-4,
    min_lr          = 1e-6,
    weight_decay    = 5e-4,
    device          = "cuda",
    lr_patience     = 10,
    lr_factor       = 0.8,
    bs_phase1       = 2048,
    bs_phase2       = 512,
    bs_phase3       = 256,
    bs_phase4       = 128,
    lr_mult_phase1  = 4.0,
    lr_mult_phase2  = 1.0,
    lr_mult_phase3  = 0.5,
    lr_mult_phase4  = 0.25,
    num_classes     = 1,
    d_model         = 512,
    d_ff            = 2048,
    n_heads         = 16,
    num_layers      = 8,
    num_freqs       = 16,
    dropout         = 0.15,
)

train_with_meanp_gate0_reactive(args)
```

### Make predictions

```bash
python /content/make_gap_value_prediction.py \
    --dataset_path    "/path/test_graphs.pt" \
    --checkpoint_path "/path/ogb_meanp_reactive/chemgraphormer_best_model.pt" \
    --output_dir      "/path/predictions" \
    --device          "cuda" \
    --split           "test"
```

---

## Best model weights

> **[All best model weights can be downloaded or retrieved via the google link `https://drive.google.com/drive/folders/1ioIc7KZNwoHA_AD-8ImEc5Tkp2h5bJb8?usp=sharing`]**

| Checkpoint | Task | Val MAE | Parameters |
|---|---|---|---|
| `chemgraphormer_gate_zero_ablation_best_model.pt` | OGB PCQM4Mv2 | 0.1013 eV | 23,643,649 |
| `chemgraphormer_best_model.pt` | QM9 ZPVE | 8.75 meV | 11,336,673 |

---

## Citation

```bibtex
@article{chemgraphormer2026,
  title   = {},
  author  = {},
  journal = {},
  volume  = {},
  pages   = {},
  year    = {2026},
  doi     = {},
}
```

---

## Acknowledgements

```
Department of Biomedical Engineering, University of Ghana.
Supervisors: Prof. Samuel Kojo Kwofie.  |  Dr. Claude Fiifi Hayford.
Experiments: Google Colab · NVIDIA L4 GPU (22.5 GB).
```

**License:** MIT
