"""
Update an existing W&B artifact's description within a W&B run.

This code initializes a W&B run, retrieves the specified artifact by name and alias,
updates its description, and saves the changes.
"""
import wandb

with wandb.init(entity="<entity>", project="<project>") as run:
    # Retrieve the artifact by name and alias
    artifact = run.use_artifact(artifact_or_name="<artifact>:<alias>")
    # Update the artifact's description
    artifact.description = "<description>"
    # Save the updated artifact
    artifact.save()
