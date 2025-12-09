"""
Add a tag to an artifact when logging it to W&B.
"""

import wandb

# Create an artifact
artifact = wandb.Artifact(name="my-artifact", type="dataset")

# Log the artifact with tags
with wandb.init(project="<project>") as run:
    run.log_artifact(artifact, tags=["tag1", "tag2"])