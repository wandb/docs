**Valid metric names:**
```python
# Valid metric names
wandb.log({"accuracy": 0.9, "val_loss": 0.1, "epoch_5": 5})
wandb.log({"modelAccuracy": 0.95, "learning_rate": 0.001})
```

**Invalid metric names (avoid these):**
```python
wandb.log({"acc,val": 0.9})  # Contains comma
wandb.log({"loss-train": 0.1})  # Contains hyphen
wandb.log({"test acc": 0.95})  # Contains space
wandb.log({"5_fold_cv": 0.8})  # Starts with number
```