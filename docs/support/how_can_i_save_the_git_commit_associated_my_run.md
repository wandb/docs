---
title: "How can I save the git commit associated with my run?"
tags: []
---

### How can I save the git commit associated with my run?
When `wandb.init` is called in your script, we automatically look for git information to save, including a link to a remote repo and the SHA of the latest commit. The git information should show up on your [run page](../app/pages/run-page.md). If you aren't seeing it appear there, make sure that your shell's current working directory when executing your script is located in a folder managed by git.

The git commit and command used to run the experiment are visible to you but are hidden to external users, so if you have a public project, these details will remain private.