"""
Add one or more tags to previously saved runs.

Use the Public API to update tags on stored data after the run has finished or
when you are working outside the run process.

To add a new tag to a run, update the "tags"
property with a new list that includes the existing tags and the new tag(s).
The run's path consists of entity/project/run_id
After updating the tags, call the run's `update()` method to save the changes.
"""
import wandb

entity = "<entity>"
project = "<project>"
run_id = "<run-id>"  # Replace with the ID of the run you want to update

# Path consists of entity/project/run_id
with wandb.Api().run(f"{entity}/{project}/{run_id}") as run:
  run.tags.append("<tag>")
  run.update()