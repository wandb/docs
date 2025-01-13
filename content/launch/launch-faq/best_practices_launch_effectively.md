---
menu:
  launch:
    identifier: best_practices_launch_effectively
    parent: launch-faq
title: Are there best practices for using Launch effectively?
---

1. Create the queue before starting the agent to enable easy configuration. Failure to do this results in errors that prevent the agent from functioning until a queue is added.

2. Create a W&B service account to initiate the agent, ensuring it is not linked to an individual user account.

3. Use `wandb.config` to manage hyperparameters, allowing for overwriting during job re-runs. Refer to [this guide](/guides/track/config/#set-the-configuration-with-argparse) for details on using argparse.