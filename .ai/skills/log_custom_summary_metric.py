"""
Log a custom summary metric to W&B.
"""
import wandb
import random 

with wandb.init(project="<project>") as run:
    # Log a custom summary metric with a random integer value between 1 and 10
    run.summary["<metric_name>"] = random.randint(1, 10)