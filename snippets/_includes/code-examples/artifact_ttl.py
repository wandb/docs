"""
Create and log an artifact with a TTL policy in W&B.
"""
import wandb
from datetime import timedelta

# Create an artifact with TTL policy
artifact = wandb.Artifact(name="<artifact_name>", type="<artifact_type>")
artifact.add_file("<file_path>")
artifact.ttl = timedelta(days=30)  # Set TTL policy

with wandb.init(project="<project>", entity="<entity>") as run:
    # Log the artifact with TTL
    run.log_artifact(artifact)