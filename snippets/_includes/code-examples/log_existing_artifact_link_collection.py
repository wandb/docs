"""
Download and log an existing artifact from a W&B registry collection.
Replace the placeholders with actual registry, collection names, entity,
project, and version.
"""
import wandb

# Construct the full artifact name with version
registry_name = "<registry_name>"  # Specify the registry name
collection_name = "<collection_name>" # Specify the collection name
version = 0 # Specify the version of the artifact to use
artifact_name_registry = f"wandb-registry-{registry_name}/{collection_name}:v{version}"

# Initialize a W&B run in the different team and project
with wandb.init(entity="<entity>", project="<project>") as run:
    # Use the model artifact from the registry
    registry_model = run.use_artifact(artifact_or_name=artifact_name_registry)

    # Download the model to a local directory
    local_model_path = registry_model.download()
