---
title: How do I log to the right wandb user on a shared machine?
displayed_sidebar: support
tags:
- logs
---
When using a shared machine, ensure that runs log to the correct WandB account by setting the `WANDB_API_KEY` environment variable for authentication. If sourced in the environment, this variable provides the correct credentials upon login. Alternatively, set the environment variable directly in the script.

Execute the command `export WANDB_API_KEY=X`, replacing X with your API key. Logged-in users can find their API key at [wandb.ai/authorize](https://app.wandb.ai/authorize).