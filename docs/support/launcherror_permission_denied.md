---
title: How do I fix a "permission denied" error in Launch?
displayed_sidebar: launch
---
If you encounter the error message `Launch Error: Permission denied`, it indicates insufficient permissions to log to the desired project. Possible causes include:

1. You are not logged in on this machine. Run [`wandb login`](../ref/cli/wandb-login.md) in the command line.
2. The specified entity does not exist. The entity must be your username or an existing team's name. Create a team if necessary with the [Subscriptions page](https://app.wandb.ai/billing).
3. You lack project permissions. Request the project creator to change the privacy setting to **Open** to allow logging runs to the project.