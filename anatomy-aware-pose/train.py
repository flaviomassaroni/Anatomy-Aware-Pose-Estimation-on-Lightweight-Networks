"""Train: loss pesata, ciclo di training, checkpointing.
Sezione 'Train' della struttura richiesta dalla prof.
"""
import os
import time
import csv
import torch
import torch.nn as nn
from tqdm.auto import tqdm


class WeightedMSELoss(nn.Module):
    """MSE sulle heatmap, pesata per keypoint (i keypoint con v=0 non contano)."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, target_weight):
        B, K, H, W = pred.shape
        pred = pred.reshape(B, K, -1)
        target = target.reshape(B, K, -1)
        loss = self.mse(pred, target).mean(dim=-1)      # [B, K]
        loss = loss * target_weight.squeeze(-1)         # azzera i keypoint non validi
        return loss.sum() / (target_weight.sum() + 1e-6)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running = 0.0
    for imgs, hms, weights in tqdm(loader, desc="train", leave=False):
        imgs, hms, weights = imgs.to(device), hms.to(device), weights.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, hms, weights)
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running = 0.0
    for imgs, hms, weights in tqdm(loader, desc="val", leave=False):
        imgs, hms, weights = imgs.to(device), hms.to(device), weights.to(device)
        out = model(imgs)
        running += criterion(out, hms, weights).item() * imgs.size(0)
    return running / len(loader.dataset)


def save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch'], ckpt['val_loss']


def fit(model, train_loader, val_loader, optimizer, scheduler, criterion,
        device, num_epochs, checkpoint_dir, resume=True):
    """Ciclo completo con resume da last.pth, salvataggio best.pth e history.csv."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    history = []
    start_epoch = 1
    last_path = f'{checkpoint_dir}/last.pth'

    if resume and os.path.exists(last_path):
        start_epoch, best_val_loss = load_checkpoint(last_path, model, optimizer, scheduler, device)
        start_epoch += 1
        print(f"Checkpoint trovato: riprendo da epoca {start_epoch} (best finora: {best_val_loss:.6f})")
    else:
        print("Nessun checkpoint: training da zero")

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{num_epochs} | train: {train_loss:.6f} | "
              f"val: {val_loss:.6f} | lr: {lr_now:.1e} | {time.time() - t0:.1f}s")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': lr_now})
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{checkpoint_dir}/best.pth')
            print(f"  -> nuovo best (val_loss: {val_loss:.6f})")

    with open(f'{checkpoint_dir}/history.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'lr'])
        writer.writeheader()
        writer.writerows(history)
    return history
