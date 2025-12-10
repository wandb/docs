"""
Add an description to a collection in a registry.
"""
import wandb

# Initialize W&B API
api = wandb.Api()

# Define registry and collection details
collection_type = "<collection_type>"
registry_name = "<registry_name>"
collection_name = "<collection_name>"

# Construct the full registry path
registry_path = f"wandb-registry-{registry_name}/{collection_name}"

# Retrieve the artifact collection
collection = api.artifact_collection(
  type_name = collection_type, 
  name = registry_path
  )

# Add description annotation to the collection object
collection.description = "<description>"

# Save the updated collection
collection.save()