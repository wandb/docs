---
title: トレーニングコードで run の名前をどのように設定できますか？
menu:
  support:
    identifier: ja-support-kb-articles-configure_name_run_training_code
support:
  - experiments
toc_hide: true
type: docs
url: /ja/support/:filename
---
トレーニングスクリプトの最初に、実験名を指定して `wandb.init` を呼び出します。例: `wandb.init(name="my_awesome_run")`.