---
title: If wandb crashes, will it possibly crash my training run?
menu:
  support:
    identifier: ja-support-crashes_crash_training_run
tags:
- crashing and hanging runs
toc_hide: true
type: docs
---

トレーニング の run との干渉を避けることは重要です。 W&B は別の プロセス で動作し、W&B でクラッシュが発生した場合でも、 トレーニング が継続されるようにします。インターネットが停止した場合、W&B は [wandb.ai](https://wandb.ai) へのデータ送信を継続的に再試行します。
