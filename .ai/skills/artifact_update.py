"""
Update an existing W&B artifact's description within a W&B run.

This code initializes a W&B run, retrieves the specified artifact by name and alias,
updates its description, and saves the changes.
"""
import wandb

with wandb.init(entity="<entity>", project="<project>") as run:
    artifact = run.use_artifact(artifact_or_name="<artifact>:<alias>")
    artifact.description = "<description>"
    artifact.save()