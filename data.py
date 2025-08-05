import polars as pl
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from dbg import dprint, tstats

# preprocess
# replace non-ATGCN characters with 'N' and ensure length is 8192

def load_and_process(data_path="data"):
    """minimal preprocessing of sequences and labels.
    
    for sequences:
    - replace non-ATGCN characters with 'N'
    - ensure length is 8192 by padding with 'N' if shorter or truncating
    for labels:
    - melt the labels
    """
    data_path = Path(data_path)
    seqs = pl.read_csv(data_path / "train_values.csv")
    labs = pl.read_csv(data_path / "train_labels.csv")

    # preprocess sequences
    seqs = seqs.with_columns(
        pl.col("sequence")
        .str.replace_all(r"[^ATGCN]", "N")
        # ensure length is 8192
        .str.slice(0, 8192)  
        .str.pad_end(8192, "N")  
    )
    # preprocess labels
    labs = (labs.unpivot(index=["sequence_id"])
        .filter(pl.col("value") == 1.0)
        .select([
            pl.col("sequence_id"),
            pl.col("variable").alias("lab")
        ]))
    return seqs, labs

def split(data: pl.DataFrame, fraction: float = 0.8):
    """
    Split the data into training and testing sets.
    """
    shuffled = data.sample(fraction=1.0, shuffle=True, with_replacement=False)
    split_idx = int(len(shuffled) * fraction)
    train_data = shuffled[:split_idx]
    test_data = shuffled[split_idx:]
    return train_data, test_data

def get_dataloaders(batch_size, split_fraction=0.8, data_path="data"):
    """
    Load and process data, split into train and test sets, and create DataLoaders.
    """
    vals, labs = load_and_process(data_path)

    train_vals, test_vals = split(vals, split_fraction)
    train_labs, test_labs = split(labs, split_fraction)
    
    dprint(1, f"Train size: {len(train_vals)}, Test size: {len(test_vals)}")
      
    trainset = NTSDataset(
        seqs=train_vals['sequence'].to_list(),
        labs=train_labs['lab'].to_list()
    )
    testset = NTSDataset(
        seqs=test_vals['sequence'].to_list(),
        labs=test_labs['lab'].to_list()
    )
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    classes = sorted(set(train_labs['lab'].to_list()))
    dprint(1, f"Classes: {classes[:10]}... Total: {len(classes)}")
    
    return trainloader, testloader
    

class NTSDataset(Dataset):
    def __init__(self, seqs: list, labs: list):
        self.seqs = seqs
        self.labs = labs
        self.nt_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.lab_to_idx = {lab: i for i, lab in enumerate(sorted(set(self.labs)))}
        
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        idxs = torch.tensor([self.nt_to_idx.get(nt, 4) for nt in seq], dtype=torch.long)
        one_hot = F.one_hot(idxs, num_classes=len(self.nt_to_idx)).float()
        one_hot = torch.permute(one_hot, (1, 0)) # torch expects (channels, sequence_length)
        label = self.lab_to_idx[self.labs[idx]]
        return one_hot, label


if __name__ == "__main__":
    trainloader, testloader = get_dataloaders(batch_size=16, split_fraction=0.8, data_path="data")
    # classes = sorted(set(train_labs['lab'].to_list()))
    # dprint(1, f"Classes: {classes[:10]}... Total: {len(classes)}")
    
    dprint(1, len(trainloader), "training batches, len(testloader)", len(testloader))