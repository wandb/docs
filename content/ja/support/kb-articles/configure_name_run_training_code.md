---
title: トレーニング コード内で run の名前を設定するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-configure_name_run_training_code
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

トレーニングスクリプトの冒頭で、`wandb.init` を使って experiment の名前を指定してください。例えば：`wandb.init(name="my_awesome_run")`。