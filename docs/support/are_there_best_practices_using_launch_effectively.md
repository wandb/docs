---
title: "Are there best practices for using Launch effectively?"
tags:
   - launch
---

1. Create your queue before you start your agent, so that you can set your agent to point to it easily.  If you don’t do this, your agent will give errors and not work until you add a queue.
  2. Create a W&B service account to start up the agent, so that it's not tied to an individual user account.
  3. Use `wandb.config` to read and write your hyperparameters, so that they can be overwritten when re-running a job. Check out [this guide](/guides/track/config/#set-the-configuration-with-argparse) if you use argsparse.