# ChemGraphormer

**A Bond-Aware Message-Passing Sparse Graph Attention Transformer for Molecular Property Prediction**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Overview

ChemGraphormer is a chemically inductive sparse graph attention Transformer that learns molecular representations exclusively from **2D molecular graphs**: no 3D coordinates, bond lengths, or conformers required. It introduces sparse bond-restricted dot-product attention computed only over directly bonded atom pairs via `scatter_softmax` / `scatter_add`, with triple simultaneous bond conditioning of the attention logit, value content, and message amplitude within a single unified attention operation.

**QM9 ZPVE results (standard benchmark split adapted from DimeNet — 110k/10k/10,831):**

| Model | ZPVE MAE on 10831 (meV) | 3D? |
|---|---|---|
| **ChemGraphormer (Mean Pool + Gate b=0)** | **8.75** | **No** |
| **ChemGraphormer (Mean Pool + Gate b=1)** | **8.37** | **No** |
| DimeNet++ | 1.210 | Yes |
| PaiNN | 1.280 | Yes |
| SchNet | 1.700 | Yes |

---

## 1. Clone Repository

**PLEASE NOTE: All experiments were conducted using Python 3.12**

```bash
git clone https://github.com/gkd-labs/ChemGraphormer.git
cd ChemGraphormer
```

---

## 2. Install Requirements

This repo uses **three** requirements files for three different environments:

- **`requirements.txt`** — for training (GPU/CUDA-enabled machine).
- **`graph_requirements_cpu_only.txt`** — for graph data computation on a CPU-only machine (exact pinned versions for `torch` + `dgl` + `torchdata`).
- **`graph_requirements_gpu_avaialble.txt`** — for graph data computation on a machine with a CUDA-capable GPU available.

Both graph-computation requirements files currently **cannot run in Google Colab** — there is no DGL wheel there compatible with any available torch and python version, on either CPU or GPU. Run graph computation in a local terminal or virtual machine instead.

```bash
# RUN ONLY ONE OF THESE TWO DEPENDING ON THE MACHINE YOU ARE USING, WHETHER CPU ONLY OR WITH GPU AVAILABLE


# On your graph-computation machine with only CPU available (terminal, not Colab compatible)
pip install -r graph_requirements_cpu_only.txt

# On your graph-computation machine with available GPU (terminal or virtual machine, not Colab compatible) 
pip install -r graph_requirements_gpu_avaialble.txt
```

```bash
# On your training machine (GPU) - Colab compatible
pip install -r requirements.txt
```

---

## 3. QM9 Dataset Preparation

Download QM9 from HuggingFace, remove uncharacterized molecules, and split into the standard benchmark splits (adopted from DimeNet — 110,000 : 10,000 : 10,831) in one step:

```bash
python load_qm9_130831.py
```

This saves four CSV files to the working directory:

- `qm9_130831.csv` — full cleaned dataset (130,831 molecules)
- `qm9_train.csv` — 110,000 molecules
- `qm9_val.csv` — 10,000 molecules
- `qm9_test.csv` — 10,831 molecules

Each split CSV has `smiles` and `zero_point_energy` columns, ready for Section 4.

---

## 4. Graph Data Computation

> **Note:** Run this section in a terminal with `graph_requirements_cpu_only.txt` (CPU-only machine) or `graph_requirements_gpu_avaialble.txt` (GPU-available machine) installed, see Section 2. Graph computation currently **cannot run in Google Colab**, on either CPU or GPU, since there is no DGL wheel there compatible with any available torch and python version. Pass the exact Input SMILES CSVs column named to `--smiles-col` or will default to `smiles`.
> `utils/graph_generator.py` handles all graph computations. Only set `--k` to the max heavy-atom (node) Laplacian positional encoding (PE) dimension: **k = 9** for QM9 ablation, **k = 51** for OGB pretraining.

### Compute graphs

```bash
# QM9 ablation (k = 9)
python utils/graph_generator.py \
    --input qm9_train.csv \
    --smiles-col smiles \
    --k 9 \
    --output graph_data_train.pt

# OGB PCQM4Mv2 (k = 51)
python utils/graph_generator.py \
    --input train_dataset.csv \
    --smiles-col smiles \
    --k 51 \
    --output graph_data_train.pt
```

Repeat for each split (`--input qm9_val.csv`, `qm9_test.csv`, etc.), pointing `--output` to a distinct `.pt` file per split.

### Attach labels

```bash
# QM9 ablation
python utils/add_graph_labels.py \
    --graphs graph_data_train.pt \
    --labels-csv qm9_train.csv \
    --label-col zero_point_energy \
    --output train_graphs.pt

# OGB PCQM4Mv2
python utils/add_graph_labels.py \
    --graphs graph_data_train.pt \
    --labels-csv train_dataset.csv \
    --label-col gap_val \
    --output train_graphs.pt
```

This saves a list of `torch_geometric.data.Data` objects (`x`, `edge_index`, `edge_attr`, `y`, `lap_pos`) to the `--output` path, ready for Section 5/7 training. Repeat for each split.

---

## 5. Ablation Training

The example below runs **Gate Init Zero** (Group A). To run any other condition, swap in the corresponding script from the table below and update `--ckpt_dir` and `--epoch_log_path` accordingly.

| Condition | Script |
|---|---|
| Gate b=0 (full model) | `utils/chemgraphormer_ablate_gate_zero_training_pipeline.py` |
| Gate b=1 (full model) | `utils/chemgraphormer_ablate_gate_one_training_pipeline.py` |
| Gate b=2 (full model) | `utils/chemgraphormer_ablate_gate_two_training_pipeline.py` |
| No edge bias | `utils/chemgraphormer_ablate_edge_training_pipeline.py` |
| No ΔRPE bias | `utils/chemgraphormer_ablate_rpe_training_pipeline.py` |
| No edge bias + no ΔRPE | `utils/chemgraphormer_ablate_rpe_edge_training_pipeline.py` |
| No sinusoidal RPE | `utils/chemgraphormer_ablate_no_sinusoidal_rpe_training_pipeline.py` |
| No edge msg + no gate | `utils/chemgraphormer_ablate_no_edge_msg_and_gate_training_pipeline.py` |
| Static edge flow (no gate) | `utils/chemgraphormer_ablate_static_edge_flow_training_pipeline.py` |
| Mean pool + gate b=0 | `utils/chemgraphormer_ablate_use_mean_pooling_gate_zero_training_pipeline.py` |
| Mean pool + gate b=1 | `utils/chemgraphormer_ablate_use_mean_pooling_gate_one_training_pipeline.py` |
| Mean pool + gate b=2 | `utils/chemgraphormer_ablate_use_mean_pooling_gate_two_training_pipeline.py` |

```bash
python utils/chemgraphormer_ablate_gate_zero_training_pipeline.py \
    --train_path              train_graphs.pt \
    --valid_path               valid_graphs.pt \
    --ckpt_dir                 gate0/ \
    --epoch_log_path           gate0/log_gate0.csv \
    --num_classes               1 \
    --d_model                   512 \
    --d_ff                      1024 \
    --num_freqs                 8 \
    --n_heads                   8 \
    --num_layers                6 \
    --epochs                    200 \
    --warmup_epochs              10 \
    --batch_size                128 \
    --lr                        8e-5 \
    --min_lr                    1e-6 \
    --weight_decay               5e-4 \
    --early_stopping_patience    40 \
    --dropout                   0.15 \
    --device                    cuda
```

> `--device` accepts `cuda` or `cpu` — if `cuda` is requested but unavailable, the script falls back to CPU automatically with a warning.

---

## 6. Test Set Prediction and Convergence Efficiency

### Make predictions on test set

```bash
python utils/predict_zpve_values_attnpl.py \
    --dataset_path    test_graph_data.pt \
    --checkpoint_path gate0/chemgraphormer_gate_zero_ablation_best_model.pt \
    --variant         gate_zero \
    --num_classes     1 \
    --d_model         512 \
    --n_heads         8 \
    --num_layers      6 \
    --d_ff            1024 \
    --num_freqs       8 \
    --dropout         0.0 \
    --batch_size      128 \
    --device          cuda \
    --output_dir      predictions/ \
    --split           test
```

This prints the dataset size and, if the dataset has labels, the test MAE, and saves predictions to `predictions/test_predictions.csv`.

`--variant` picks the correct model architecture for the checkpoint — it must match how the checkpoint was trained:

| `--variant` | Trained by |
|---|---|
| `gate_zero` | `utils/chemgraphormer_ablate_gate_zero_training_pipeline.py` |
| `gate_one` | `utils/chemgraphormer_ablate_gate_one_training_pipeline.py` |
| `gate_two` | `utils/chemgraphormer_ablate_gate_two_training_pipeline.py` |
| `edge` | `utils/chemgraphormer_ablate_edge_training_pipeline.py` |
| `rpe` | `utils/chemgraphormer_ablate_rpe_training_pipeline.py` |
| `rpe_edge` | `utils/chemgraphormer_ablate_rpe_edge_training_pipeline.py` |
| `no_sinusoidal_rpe` | `utils/chemgraphormer_ablate_no_sinusoidal_rpe_training_pipeline.py` |
| `no_edge_msg_and_gate` | `utils/chemgraphormer_ablate_no_edge_msg_and_gate_training_pipeline.py` |
| `static_edge_flow` | `utils/chemgraphormer_ablate_static_edge_flow_training_pipeline.py` |

> **Important:** For mean pooling ablation variants (Group D), use `utils/predict_zpve_values_meanpl.py` instead, with `--variant` set to `mean_pool_gate_zero`, `mean_pool_gate_one`, or `mean_pool_gate_two` — all other arguments remain identical:

```bash
python utils/predict_zpve_values_meanpl.py \
    --dataset_path    test_graph_data.pt \
    --checkpoint_path meanpool_gate0/chemgraphormer_mean_pooling_gate_zero_ablation_best_model.pt \
    --variant         mean_pool_gate_zero \
    --output_dir      predictions/ \
    --split           test
```

### Compute convergence efficiency

```bash
python utils/compute_convergence_efficiency.py --log_path gate0/log_gate0.csv
```

---

## 7. OGB PCQM4Mv2 training

Note: Input SMILES must be a Python DataFrame with a column named such as "smiles".
Graph computation for OGB follows the same steps as Section 4, using `utils/graph_generator.py` with `--k 51`.

### Run training

```bash
python utils/train_chemgraphormer_meanp_reactive.py \
    --train_path       train_graphs.pt \
    --valid_path       valid_graphs.pt \
    --ckpt_dir         ogb_meanp_reactive/ \
    --epoch_log_path   ogb_meanp_reactive/train_log.csv \
    --epochs           150 \
    --batch_size       1024 \
    --lr               1e-4 \
    --min_lr           1e-6 \
    --weight_decay     5e-4 \
    --device           cuda \
    --lr_patience      10 \
    --lr_factor        0.8 \
    --bs_phase1        2048 \
    --bs_phase2        512 \
    --bs_phase3        256 \
    --bs_phase4        128 \
    --lr_mult_phase1   4.0 \
    --lr_mult_phase2   1.0 \
    --lr_mult_phase3   0.5 \
    --lr_mult_phase4   0.25 \
    --num_classes      1 \
    --d_model          512 \
    --d_ff             2048 \
    --n_heads          16 \
    --num_layers       8 \
    --num_freqs        16 \
    --dropout          0.15
```

### Make predictions

```bash
python utils/make_gap_value_prediction.py \
    --dataset_path    test_graphs.pt \
    --checkpoint_path ogb_meanp_reactive/chemgraphormer_best_model.pt \
    --output_dir      predictions/ \
    --device          cuda \
    --split           test
```

---

## Best model weights

> **[All best model weights can be downloaded or retrieved via the google link `https://drive.google.com/drive/folders/1ioIc7KZNwoHA_AD-8ImEc5Tkp2h5bJb8?usp=sharing`]**

## Summary Checkpoints and Performances
| Checkpoint | Task | Val MAE | Test MAE | Parameters |
|---|---|---|---|
| `chemgraphormer_best_model.pt` | OGB PCQM4Mv2 | 0.1013 eV | - | 23,643,649 |
| `chemgraphormer_gate_zero_ablation_best_model.pt` | QM9 ZPVE | 13.33 meV | 14.60 meV | 11,336,673 |
| `chemgraphormer_gate_one_ablation_best_model.pt` | QM9 ZPVE | 13.88 meV | 14.15 meV | 11,336,673 |
| `chemgraphormer_gate_two_ablation_best_model.pt` | QM9 ZPVE | 14.15 meV | 14.55 meV | 11,336,673 |
| `chemgraphormer_edge_ablation_best_model.pt` | QM9 ZPVE | 13.06 meV | 13.48 meV | 11,336,673 |
| `chemgraphormer_rpe_ablation_best_model.pt` | QM9 ZPVE | 13.33 meV | 13.43 meV | 11,336,673 |
| `chemgraphormer_rpe_edge_ablation_best_model.pt` | QM9 ZPVE | 14.15 meV | 15.52 meV | 11,336,673 |
| `chemgraphormer_no_edge_msg_ablation_best_model.pt` | QM9 ZPVE | 14.42 meV | 33.56 meV | 11,336,673 |
| `chemgraphormer_no_sinusoidal_rpe_ablation_best_model.pt` | QM9 ZPVE | 13.06 meV | 13.89 meV | 11,336,673 |
| `chemgraphormer_static_edge_flow_ablation_best_model.pt` | QM9 ZPVE | 14.15 meV | 25.99.43 meV | 11,336,673 |
| `chemgraphormer_mean_pooling_gate_zero_ablation_best_model.pt` | QM9 ZPVE | 8.71 meV | 8.75 meV | 11,336,673 |
| `chemgraphormer_mean_pooling_gate_one_ablation_best_model.pt` | QM9 ZPVE | 8.98 meV | 9.37 meV | 11,336,673 |
| `chemgraphormer_mean_pooling_gate_two_ablation_best_model.pt` | QM9 ZPVE | 8.98 meV | 9.49 meV | 11,336,673 |

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
