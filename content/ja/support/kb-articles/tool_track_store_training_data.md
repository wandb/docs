---
title: あなたのツールはトレーニングデータを追跡または保存しますか？
menu:
  support:
    identifier: ja-support-kb-articles-tool_track_store_training_data
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

`wandb .config.update(...)` に SHA または一意の識別子を渡して、データセットを トレーニング run と関連付けます。W&B は、ローカルファイル名で `wandb .save` が呼び出されない限りデータ を保存しません。