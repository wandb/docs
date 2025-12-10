"""
Update the TTL policy of an existing artifact in W&B.
"""
import wandb
from datetime import timedelta

api = wandb.Api()

# Retrieve the existing artifact
artifact = api.artifact("<entity/project/artifact:alias>")
artifact.ttl = timedelta(days=365)  # Delete in one year
artifact.save()