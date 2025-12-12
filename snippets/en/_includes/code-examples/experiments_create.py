"""
Create an experiment in W&B. If the project does not exist, W&B creates it.

Note that this file only initializes the experiment; you can add code to log metrics,
artifacts, etc., within the `with` block.
"""
import wandb

# Initialize a W&B run
with wandb.init(project="<project>") as run:
    # Experiment code goes here
    pass
