"""
Downloads specific files or sub-folders from W&B artifacts.
"""
import wandb

with wandb.init(project="<project>") as run:
    # Indicate the artifact to use. Format is "name:alias"
    artifact = run.use_artifact("<artifact_name>:<alias>")

    # Download a specific file or sub-folder
    artifact.download(path_prefix="<file_name>") # downloads only the specified file or folder