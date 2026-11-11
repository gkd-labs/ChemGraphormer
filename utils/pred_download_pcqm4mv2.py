# ---------------------------------------------------------
# Patch torch.load so ogb's internal split-loading works
# under PyTorch >=2.6's new weights_only=True default.
# Safe here because split_dict.pt comes from OGB's own
# trusted servers.
# ---------------------------------------------------------
import torch
from ogb.lsc import PCQM4Mv2Dataset
import pandas as pd

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def prepare_pcqm4mv2_data(root_path: str):
    """
    Orchestrates the download, splitting, index resetting, removal of samples
    with nan gap values, and deduplication for the OGB PCQM4Mv2 dataset.

    Args:
        root_path (str): The root directory where the dataset will be stored.

    Returns:
        tuple: A tuple containing four pandas DataFrames:
               (train_dataset, valid_dataset, testdev_dataset, testchallenge_dataset)
               Each DataFrame contains 'smiles' and 'gap_val' columns.
    """
    pcq = PCQM4Mv2Dataset(root=root_path, only_smiles=True)

    smiles = []
    gap_val = []
    for i in range(len(pcq)):
        smiles.append(pcq[i][0])
        gap_val.append(pcq[i][1])

    pc_df = pd.DataFrame({'smiles': smiles, 'gap_val': gap_val})

    split_dict = pcq.get_idx_split()
    train_idx = split_dict['train']
    valid_idx = split_dict['valid']
    testdev_idx = split_dict['test-dev']
    testchallenge_idx = split_dict['test-challenge']

    train_dataset = pc_df.iloc[train_idx].reset_index(drop=True)
    valid_dataset = pc_df.iloc[valid_idx].reset_index(drop=True)
    testdev_dataset = pc_df.iloc[testdev_idx].reset_index(drop=True)
    testchallenge_dataset = pc_df.iloc[testchallenge_idx].reset_index(drop=True)

    train_dataset = train_dataset.dropna(subset=['gap_val']).reset_index(drop=True)
    valid_dataset = valid_dataset.dropna(subset=['gap_val']).reset_index(drop=True)

    train_dataset = train_dataset.drop_duplicates(subset=['smiles']).reset_index(drop=True)
    valid_dataset = valid_dataset.drop_duplicates(subset=['smiles']).reset_index(drop=True)

    print("train:", train_dataset.shape)
    print("valid:", valid_dataset.shape)
    print("test-dev:", testdev_dataset.shape)
    print("test-challenge:", testchallenge_dataset.shape)

    return train_dataset, valid_dataset, testdev_dataset, testchallenge_dataset


if __name__ == "__main__":
    train_dataset, valid_dataset, testdev_dataset, testchallenge_dataset = prepare_pcqm4mv2_data('./')

    # Save each split to disk so the work isn't lost when the process exits
    train_dataset.to_csv('train_dataset.csv', index=False)
    valid_dataset.to_csv('valid_dataset.csv', index=False)
    testdev_dataset.to_csv('testdev_dataset.csv', index=False)
    testchallenge_dataset.to_csv('testchallenge_dataset.csv', index=False)
    print("Saved all splits to CSV in the current directory.")