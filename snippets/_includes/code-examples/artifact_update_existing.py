"""
Given an existing artifact, update its description, metadata, and aliases
without creating a new run.
"""
import wandb

api = wandb.Api()

artifact = api.artifact(name="<entity/project/artifact:alias>")

# Update the description
artifact.description = "My new description"

# Selectively update metadata keys
artifact.metadata["oldKey"] = "new value"

# Replace the metadata entirely
artifact.metadata = {"newKey": "new value"}

# Add an alias
artifact.aliases.append("best")

# Remove an alias
artifact.aliases.remove("latest")

# Completely replace the aliases
artifact.aliases = ["replaced"]

# Persist all artifact modifications
artifact.save()