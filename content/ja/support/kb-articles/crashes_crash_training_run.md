---
title: If wandb crashes, will it possibly crash my training run?
menu:
  support:
    identifier: ja-support-kb-articles-crashes_crash_training_run
support:
- crashing and hanging runs
toc_hide: true
type: docs
url: /support/:filename
---

トレーニング run との干渉を避けることが重要です。 W&B は別のプロセスで動作し、W&B にクラッシュが発生した場合でも、トレーニングが継続されるようにします。インターネットが停止した場合、W&B は [wandb.ai](https://wandb.ai) へのデータ送信を継続的に再試行します。
