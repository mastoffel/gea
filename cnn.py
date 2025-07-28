import polars as pl
from dbg import dprint, tstats
import torch

train_vals = pl.read_csv("data/train_values_processed.csv")
train_labs = pl.read_csv("data/train_labels_processed.csv")

dprint(1, f"vals: {train_vals.shape}, labs: {train_labs.shape}")

nts = ['A', 'T', 'G', 'C', 'N']
nt_map = {nt: i for i, nt in enumerate(nts)}
dprint(1, nt_map)

x = torch.randn(10, 4, 8000) 
tstats(x, "x", level=2)