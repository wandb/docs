---
title: "How can I configure the name of the run in my training code?"
displayed_sidebar: support
tags:
   - experiments
---
At the beginning of the training script, call `wandb.init` with an experiment name. For example: `wandb.init(name="my_awesome_run")`.