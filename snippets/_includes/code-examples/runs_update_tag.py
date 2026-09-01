"""
Add one or more tags to an active run.

A run object's `tags` property is a tuple. To add a new tag to an existing run,
update the `tags` property with a new tuple that includes the existing tags and the new tag(s).
"""

import wandb

with wandb.init(entity="<entity>", project="<project>", tags=["<tag1>", "<tag2>"]) as run:
    # Training and logging code goes here

    # Add a new tag to the existing tags by creating a new tuple that includes the existing tags and the new tag.
    run.tags += ("<tag3>",)