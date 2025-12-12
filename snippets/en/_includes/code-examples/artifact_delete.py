"""
Delete specific artifact version from a W&B run. Set delete_aliaes to `True` 
if the artifact has an alias attached to it.
"""
import wandb

# Initialize W&B API
api = wandb.Api()

# Get the run by its path. Consists of <entity>/<project>/<run_path>
runs = api.run("<entity>/<project>/<run_path>")

# wandb.Api().Run.logged_artifacts() returns a list of artifact versions
# that consists of artifact name and version <artifact_name>:v<version_number>
for artifact_version in runs.logged_artifacts():
    # Index the last two characters of the artifact version name (str) that
    # consists of the version number
    if artifact_version.name[-2:] == "v"+ "<version_number>":
        artifact_version.delete(delete_aliases=True)
