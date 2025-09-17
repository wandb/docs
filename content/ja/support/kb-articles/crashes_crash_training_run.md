---
title: もし wandb がクラッシュしたら、トレーニング run もクラッシュしてしまう可能性はありますか？
menu:
  support:
    identifier: ja-support-kb-articles-crashes_crash_training_run
support:
- クラッシュやハングする runs
toc_hide: true
type: docs
url: /support/:filename
---

トレーニング run に干渉しないことが極めて重要です。W&B は別のプロセスで動作し、W&B がクラッシュしてもトレーニングが継続するようにしています。インターネット障害が発生した場合でも、W&B は [wandb.ai](https://wandb.ai) へのデータ送信を継続的に再試行します。