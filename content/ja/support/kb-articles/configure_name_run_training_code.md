---
title: トレーニング コードで run の名前をどのように設定できますか？
menu:
  support:
    identifier: ja-support-kb-articles-configure_name_run_training_code
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

トレーニングスクリプトの冒頭で、実験名を指定して `wandb.init` を呼び出します。例えば: `wandb.init(name="my_awesome_run")`。