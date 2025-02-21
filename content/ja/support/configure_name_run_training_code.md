---
title: How can I configure the name of the run in my training code?
menu:
  support:
    identifier: ja-support-configure_name_run_training_code
tags:
- experiments
toc_hide: true
type: docs
---

トレーニングスクリプトの最初で、実験名を指定して `wandb.init` を呼び出します。例えば: `wandb.init(name="my_awesome_run")`.