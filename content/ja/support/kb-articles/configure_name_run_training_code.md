---
menu:
  support:
    identifier: ja-support-kb-articles-configure_name_run_training_code
support:
- experiments
title: How can I configure the name of the run in my training code?
toc_hide: true
type: docs
url: /support/:filename
---

At the beginning of the training script, call `wandb.init` with an experiment name. For example: `wandb.init(name="my_awesome_run")`.