"""
Adds a tag to an existing W&B artifact.
"""
import wandb

# Retrieve an existing artifact and add a tag to it
artifact = wandb.Api().artifact("entity/project/artifact:version")
artifact.tags = ["new-tag"]
artifact.save()