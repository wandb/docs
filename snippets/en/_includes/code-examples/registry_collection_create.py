"""
Ceates a W&B registry collection and links an artifact to it.
"""
import wandb

# Create an artifact object
artifact = wandb.Artifact(name = "<artifact_name>", type = "<artifact_type>")

registry_name = "<registry_name>"
collection_name = "<collection_name>"
registry_path = f"wandb-registry-{registry_name}/{collection_name}"

# Initialize a run
with wandb.init(entity = "<entity>", project = "<project>") as run:

  # Link the artifact to a collection. If the collection does not exist, W&B creates it.
  run.link_artifact(artifact = artifact, target_path = registry_path)
