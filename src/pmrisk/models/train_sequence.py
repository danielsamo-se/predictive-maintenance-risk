"""Small helpers for filtering an index and training/evaluating a PyTorch model"""
from __future__ import annotations

import torch
from torch import nn
from sklearn.metrics import average_precision_score


def filter_index_by_engine_ids(index: list[dict], engine_ids: set[int]) -> list[dict]:
    return [entry for entry in index if entry["engine_id"] in engine_ids]


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return 0.0 if num_batches == 0 else total_loss / num_batches


def eval_loss(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float().view(-1, 1)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            total_loss += float(loss.item())
            num_batches += 1

    return 0.0 if num_batches == 0 else total_loss / num_batches


def predict_logits(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits = []
    all_y_true = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            
            all_logits.append(logits.cpu())
            all_y_true.append(y_batch.cpu().float().view(-1, 1))

    logits_tensor = torch.cat(all_logits, dim=0)
    y_true_tensor = torch.cat(all_y_true, dim=0)

    return logits_tensor, y_true_tensor


def eval_metrics(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> dict:
    loss = eval_loss(model, dataloader, device)
    
    logits, y_true = predict_logits(model, dataloader, device)
    
    y_true_1d = y_true.view(-1).cpu().numpy()
    logits_1d = logits.view(-1).cpu()

    scores = torch.sigmoid(logits_1d).cpu().numpy()

    pr_auc = float(average_precision_score(y_true_1d, scores))

    return {
        "loss": loss,
        "pr_auc": pr_auc,
    }


def train_sequence_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    n_epochs: int = 20,
    patience: int = 5,
    lr: float = 1e-3,
) -> dict:

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = []
    best_pr_auc = float("-inf")
    best_loss = float("inf")
    best_state_dict = {}
    best_metrics = {}
    epochs_without_improvement = 0
    
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = eval_metrics(model, val_loader, device)
        
        val_loss = val_metrics["loss"]
        val_pr_auc = val_metrics["pr_auc"]
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pr_auc": val_pr_auc,
        })
        
        is_better = False
        if val_pr_auc > best_pr_auc:
            is_better = True
        elif val_pr_auc == best_pr_auc and val_loss < best_loss:
            is_better = True
        
        if is_better:
            best_pr_auc = val_pr_auc
            best_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            break
    
    return {
        "best_state_dict": best_state_dict,
        "best_metrics": best_metrics,
        "history": history,
    }
