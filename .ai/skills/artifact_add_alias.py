"""
Add one or more aliases to an artifact when logging it to W&B.
"""
import wandb

with wandb.init(project="<project>") as run:
    artifact = wandb.Artifact(name="<artifact_name>", type="<artifact_type>")
    artifact.add_file("file.txt")
    run.log_artifact(artifact, aliases=["latest", "other-alias"])