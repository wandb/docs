---
title: My run's state is `crashed` on the UI but is still running on my machine. What
  do I do to get my data back?
menu:
  support:
    identifier: ja-support-runs_state_crashed_ui_running_machine_get_data
tags:
- experiments
toc_hide: true
type: docs
---

トレーニング 中にマシンへの接続が失われた可能性があります。 [`wandb sync [PATH_TO_RUN]`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}}) を実行して、データを復元します。 run へのパスは、実行中の Run ID と一致する `wandb` ディレクトリー内のフォルダーです。
