
import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
from dbg import dprint, tstats
from cnn import CNN
from data import get_dataloaders
from torch import nn
import wandb

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    

def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for seqs, labs in dataloader:
            seqs, labs = seqs.to(device), labs.to(device)
            
            out = model(seqs)
            loss = criterion(out, labs)
            total_loss += loss.item()
            
            _, pred = torch.max(out, 1)
            total += labs.size(0)
            correct += (pred == labs).sum().item()
            
    avg_loss = total_loss / len(dataloader)
    acc = 100 * correct / total
    return avg_loss, acc
            
train_losses = []
val_losses = []

def train_model(
    n_epochs: int, 
    model: nn.Module, 
    trainloader: DataLoader, 
    testloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    use_wandb: bool = False):
    
    if use_wandb:
        wandb.init(project="gea", config={
            "epochs": n_epochs,
            "batch_size": trainloader.batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "model": "CNN"
        })
        wandb.watch(model)
    
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        n_steps = len(trainloader)
        
        for i, (seqs, labs) in enumerate(trainloader):
            seqs, labs = seqs.to(device), labs.to(device)
            optimizer.zero_grad()
            out = model(seqs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            epoch_loss += loss.item()
            
            if i  % 100 == 99:
                last_loss = running_loss / 1000
                dprint(1, f"Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{n_steps}], Loss: {last_loss:.4f}")
                running_loss = 0.0
                
            # average loss for the epoch
        avg_train_loss = epoch_loss / len(trainloader)
        train_losses.append(avg_train_loss)
        
        # validation
        val_loss, val_accuracy = eval_model(model, testloader, criterion, device)
        val_losses.append(val_loss)
        
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
        
        dprint(1, f"Epoch [{epoch+1}/{n_epochs}] Summary:")
        dprint(1, f"  Train Loss: {avg_train_loss:.4f}")
        dprint(1, f"  Val Loss: {val_loss:.4f}")
        dprint(1, f"  Val Accuracy: {val_accuracy:.2f}%")
        dprint(1, "-" * 40)
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    device = get_device()
    dprint(1, f"Using device: {device}")
    
    trainloader, testloader = get_dataloaders(batch_size=32)
    
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    train_model(3, model, trainloader, testloader, criterion, optimizer, device)