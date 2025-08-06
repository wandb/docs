---
title: wandb がクラッシュした場合、トレーニング run もクラッシュする可能性がありますか？
menu:
  support:
    identifier: ja-support-kb-articles-crashes_crash_training_run
support:
- クラッシュやハングする run
toc_hide: true
type: docs
url: /support/:filename
---

トレーニング run への干渉を避けることは非常に重要です。W&B は別のプロセスで動作するため、W&B がクラッシュしてもトレーニングは継続されます。インターネットが切断された場合でも、W&B は [wandb.ai](https://wandb.ai) へのデータ送信を継続的に再試行します。