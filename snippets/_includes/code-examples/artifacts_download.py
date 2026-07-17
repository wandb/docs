"""
Downloads files or entire artifacts from W&B. The same
logic applies to external artifacts.
"""

import wandb    

with wandb.init(project="<project>") as run:
    # Indicate the artifact to use. Format is "name:alias"
    artifact = run.use_artifact("<artifact_name>:<alias>")

    # Downloads file from the artifact at path name
    # If artifact.add_reference() was used, returns the reference URL
    entry = artifact.get_entry("<file_name>")

    # Download the entire artifact
    datadir = artifact.download()