---
title: "How can I save the git commit associated with my run?"
tags:
   - experiments
---
When `wandb.init` is invoked, the system automatically collects git information, including the remote repository link and the SHA of the latest commit. This information appears on the [run page](../guides/app/pages/run-page.md). Ensure the current working directory when executing the script is within a git-managed folder to view this information.

The git commit and the command used to run the experiment remain visible to the user but are hidden from external users. In public projects, these details remain private.