"""
Organize runs by their job type.

Add a job type to a run by passing the `job_type` parameter to `wandb.init(job_type="")`.
"""
import wandb

entity = "<entity>"
project = "<project>"

# Creates runs and organizes them by job type.
for job_type in ["<JobType1>", "<JobType2>"]: # Replace with job type names
    # Simulate creating two runs for each job type.
    for i in range(2):
        with wandb.init(entity=entity, project=project, job_type=job_type, name=f"{job_type}_run_{i}") as run:
            # Training and logging code goes here
            pass    