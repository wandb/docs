"""
Adds an alias to an existing W&B artifact.
"""
import wandb

# Retrieve an existing artifact and add an alias to it
artifact = wandb.Api().artifact("entity/project/artifact:version")
artifact.aliases = ["new-alias"]
artifact.save()