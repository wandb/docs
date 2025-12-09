"""Initializes a W&B run and logs a metric."""
import wandb

# Note the usage of `with` statement to ensure proper resource management.
with wandb.init(project="<project>") as run:
    # Training and logging code goes here
    pass