import polars as pl
from dbg import dprint, tstats
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

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
        one_hot = torch.permute(one_hot, (1, 0)) # torch expects (channels, sequence_length)
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

trainloader = DataLoader(trainset, batch_size=16, shuffle=True, drop_last=True)
testloader = DataLoader(testset, batch_size=16, shuffle=False, drop_last=True)
classes = sorted(set(train_labs['lab'].to_list()))
dprint(1, f"Classes: {classes[:10]}... Total: {len(classes)}")

# simple CNN model
class CNN(torch.nn.Module):
    def __init__(self, num_filters=64, kernel_size=19, seq_len=8192,fc_dim=64, num_classes=1313):
        super().__init__()
        # convolutional layer
        self.conv1 = nn.Conv1d( in_channels=5, # 5 channels for A, T, G, C, N
                                out_channels=num_filters, 
                                kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(num_filters)
        
        # fully connected layers
        self.fc1 = nn.Linear(num_filters, fc_dim)
        self.bn2 = nn.BatchNorm1d(fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.shape[2]) # global pooling
        x = x.squeeze(-1)
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.fc2(x)
        return x


# training

device =   torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dprint(1, f"Using device: {device}")

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

n_epochs = 10
train_losses = []
val_losses = []

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for seqs, labels in dataloader:
            seqs = seqs.to(device)
            labels = labels.to(device)
            
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

n_steps = len(trainloader)

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    epoch_loss = 0.0
    
    for i, (seqs, labels) in enumerate(trainloader):
        seqs = seqs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        epoch_loss += loss.item()
        
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            dprint(1, f"Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{n_steps}], Loss: {last_loss:.4f}")
            running_loss = 0.0
    
    # average loss for the epoch
    avg_train_loss = epoch_loss / len(trainloader)
    train_losses.append(avg_train_loss)
    
    # validation
    val_loss, val_accuracy = evaluate_model(model, testloader, criterion, device)
    val_losses.append(val_loss)
    
    dprint(1, f"Epoch [{epoch+1}/{n_epochs}] Summary:")
    dprint(1, f"  Train Loss: {avg_train_loss:.4f}")
    dprint(1, f"  Val Loss: {val_loss:.4f}")
    dprint(1, f"  Val Accuracy: {val_accuracy:.2f}%")
    dprint(1, "-" * 40)
        