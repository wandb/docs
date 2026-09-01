"""
Add one or more aliases to an artifact when logging it to W&B.
"""
import wandb

# Create an artifact
artifact = wandb.Artifact(name="<artifact_name>", type="<artifact_type>")
# Add files to the artifact
artifact.add_file("<file_path>")

with wandb.init(project="<project>") as run:
    # Log the artifact with aliases
    run.log_artifact(artifact, aliases=["<alias1>", "<alias2>"])