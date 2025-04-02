---
title: How can I configure the name of the run in my training code?
menu:
  support:
    identifier: ja-support-kb-articles-configure_name_run_training_code
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

トレーニングスクリプトの先頭で、`wandb.init` を experiment 名とともに呼び出します。例：`wandb.init(name="my_awesome_run")`