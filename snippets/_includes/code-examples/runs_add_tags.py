"""
Add one or more tags to a run.
"""
import wandb

with wandb.init(entity="<entity>", project="<project>", tags=["<tag1>", "<tag2>"]) as run:
    # Training and logging code goes here
    pass