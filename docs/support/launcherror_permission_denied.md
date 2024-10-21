---
title: "`LaunchError: Permission denied`"
tags:
   - experiments
---

If you're getting the error message `Launch Error: Permission denied`, you don't have permissions to log to the project you're trying to send runs to. This might be for a few different reasons.

1. You aren't logged in on this machine. Run [`wandb login`](../../ref/cli/wandb-login.md) on the command line.
2. You've set an entity that doesn't exist. "Entity" should be your username or the name of an existing team. If you need to create a team, go to our [Subscriptions page](https://app.wandb.ai/billing).
3. You don't have project permissions. Ask the creator of the project to set the privacy to **Open** so you can log runs to this project.