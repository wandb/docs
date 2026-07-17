"""
Log a CSV file to W&B as a table.

Replace:
- values enclosed in angle brackets with your own
- <name>.csv in pd.read_csv() with the name of your CSV file
"""

import wandb
import pandas as pd

# Read CSV as a DataFrame object (pandas)
dataframe = pd.read_csv("<name>.csv")

# Convert the DataFrame into a W&B Table
table = wandb.Table(dataframe=dataframe)

# Start a W&B run to log data
with wandb.init(project="<project>") as run:

    # Log the table to visualize it in the W&B UI
    run.log({"<table_name>": table})