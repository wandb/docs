---
menu:
  support:
    identifier: ja-support-configure_name_run_training_code
tags:
- experiments
title: How can I configure the name of the run in my training code?
toc_hide: true
type: docs
---

At the beginning of the training script, call `wandb.init` with an experiment name. For example: `wandb.init(name="my_awesome_run")`.