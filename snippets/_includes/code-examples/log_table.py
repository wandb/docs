"""
Log a table to W&B.
"""
import wandb

# Create a table object with two columns and two rows of data
my_table = wandb.Table(
    columns=["a", "b"],
    data=[["a1", "b1"], ["a2", "b2"]],
    log_mode="<log_mode>"
    )

# Start a new run
with wandb.init(project="<project>") as run:
    # Log the table to W&B
    run.log({"<table_name>": my_table})
