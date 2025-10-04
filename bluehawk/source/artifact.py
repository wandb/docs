##### Create an artifact #####
# :snippet-start: create_artifact
import wandb

run = wandb.init(project="artifacts-example", job_type="add-dataset")
artifact = wandb.Artifact(name="example_artifact", type="dataset")
artifact.add_file(local_path="./dataset.h5", name="training_dataset")
artifact.save()
# :snippet-end: create_artifact
##### Create an artifact - END #####


##### Download an artifact #####
# :snippet-start: download_artifact-1
artifact = run.use_artifact(
    "training_dataset:latest"
)  # returns a run object using the "my_data" artifact
# :snippet-end: download_artifact-1

# :snippet-start: download_artifact-2
datadir = (
    artifact.download()
)  # downloads the full `my_data` artifact to the default directory.
# :snippet-end: download_artifact-2

##### Download an artifact - END #####