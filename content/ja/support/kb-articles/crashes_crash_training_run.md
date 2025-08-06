---
title: wandb がクラッシュした場合、トレーニング run もクラッシュする可能性がありますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- クラッシュやハングしている run
---

トレーニング run への影響を避けることは非常に重要です。W&B は別プロセスで動作しているため、W&B がクラッシュしてもトレーニングは継続されます。インターネット接続が切れた場合でも、W&B は [wandb.ai](https://wandb.ai) へのデータ送信を継続的にリトライします。