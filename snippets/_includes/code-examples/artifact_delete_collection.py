"""
Delete an artifact collection from W&B.
"""
import wandb

# Initialize W&B API
api = wandb.Api()

# Delete an artifact collection by its name and type
# Name format: <entity>/<project>/<run_path>
collection = api.artifact_collection(
    name="<entity>/<project>/<run_path>",
    type_name="<artifact_type>"
)

collection.delete()
