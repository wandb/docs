---
title: "How can I configure the name of the run in my training code?"
tags: []
---

### How can I configure the name of the run in my training code?
At the top of your training script when you call `wandb.init`, pass in an experiment name, like this: `wandb.init(name="my_awesome_run")`.