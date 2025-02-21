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

トレーニング run への干渉を避けることが重要です。W&B は別のプロセスで動作するため、W&B がクラッシュしてもトレーニングは続行されます。インターネットが途切れた場合でも、W&B は [wandb.ai](https://wandb.ai) へのデータ送信を継続的に再試行します。