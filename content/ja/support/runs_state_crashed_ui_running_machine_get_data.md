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

トレーニング中にマシンとの接続が切れた可能性があります。データを回復するには、[`wandb sync [PATH_TO_RUN]`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}}) を実行します。run へのパスは、進行中の run の Run ID と一致する `wandb` ディレクトリー内のフォルダーです。