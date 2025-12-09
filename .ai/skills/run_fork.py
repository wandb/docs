"""Fork an existing W&B run from a specific step."""

import wandb

# Initialize a run to be forked later
with wandb.init(project="<project>", entity="<entity>") as original_run:
    # Training and logging code goes here.
    pass

# Fork the run from a specific step
with wandb.init(project="<project>",entity="<entity>", fork_from=f"{original_run.id}?_step=200") as forked_run:
    # Training and logging code goes here.
    pass