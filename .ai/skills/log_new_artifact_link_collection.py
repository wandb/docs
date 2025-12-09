"""
Create a W&B artifact and link it to a collection in a registry. If the
collection does not exist, W&B creates it.
"""
import wandb

# Create an artifact object
artifact = wandb.Artifact(
  name = "<artifact_name>",
  type = "<artifact_type>"
)

# Define registry and collection names
registry_name = "<registry_name>"
collection_name = "<collection_name>"
target_path = f"wandb-registry-{registry_name}/{collection_name}"

# Initialize a run
with wandb.init(entity = "<team_entity>", project = "<project>") as run:
  # Link the artifact to a collection. If the collection does not exist, W&B creates it.
  run.link_artifact(artifact = artifact, target_path = target_path)