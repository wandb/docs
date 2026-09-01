"""
Log a scatter plot to W&B. 

Replace:
- values enclosed in angle brackets with your own
- [[1, 2], [2, 3]] in wandb.Table(data=) with your own 2D row-oriented array of values to plot

Pass a 2D row-oriented array of values to data (wandb.Table(data=)), with
column names specified in the `columns` parameter.

Column names must match the x and y parameters in the `wandb.plot.scatter()` plotting function.
"""
import wandb

# Start a new run
with wandb.init(entity="<entity>", project="<project>") as run:
        
    # Create a table with the columns to plot
    table = wandb.Table(data=[[1, 2], [2, 3]], columns=["<a_column>", "<b_column>"])

    # Use the table to populate various custom charts
    scatter = wandb.plot.scatter(table, x='<a_column>', y='<b_column>', title='<title>')

    # Log custom tables, which will show up in customizable charts in the UI
    run.log({'scatter_1': scatter})