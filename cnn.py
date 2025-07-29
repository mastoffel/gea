import polars as pl
from dbg import dprint, tstats
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

vals = pl.read_csv("data/train_values_processed.csv")
labs = pl.read_csv("data/train_labels_processed.csv")

# split into train and test 90/10 after randomizing 
vals = vals.sample(fraction=1.0, shuffle=True, with_replacement=False)
labs = labs.sample(fraction=1.0, shuffle=True, with_replacement=False)
split = int(len(vals) * 0.8)
train_vals = vals[:split]
train_labs = labs[:split]
test_vals = vals[split:]
test_labs = labs[split:]

dprint(1, f"Train size: {len(train_vals)}, Test size: {len(test_vals)}")

class NTSDataset(Dataset):
    def __init__(self, seqs: list, labs: list):
        self.seqs = seqs
        self.labs = labs
        self.nt_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.lab_to_idx = {lab: i for i, lab in enumerate(sorted(set(self.labs)))}
        
    def __len__(self):
        
        return len(self.seqs)
    
    def __getitem__(self, idx):
        # padding / truncating has been done in preprocessing
        seq = self.seqs[idx]
        idxs = torch.tensor([self.nt_to_idx.get(nt, 4) for nt in seq], dtype=torch.long)
        one_hot = F.one_hot(idxs, num_classes=len(self.nt_to_idx)).float()
        tstats(one_hot, "one_hot", level=2)
        label = self.lab_to_idx[self.labs[idx]]
        
        dprint(2, f"seq: {seq[:10]}..., label: {self.labs[idx]}")
        dprint(2, f"one_hot: {one_hot[:10]}..., label_idx: {label}")
        
        return one_hot, label

 
# create datasets and dataloaders       
trainset = NTSDataset(
    seqs=train_vals['sequence'].to_list(),
    labs=train_labs['lab'].to_list()
)
testset = NTSDataset(
    seqs=test_vals['sequence'].to_list(),
    labs=test_labs['lab'].to_list()
)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# simple CNN model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
