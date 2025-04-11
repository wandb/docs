---
title: 私の run の状態は UI 上では `crashed` ですが、マシン上ではまだ実行中です。データを取り戻すにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-runs_state_crashed_ui_running_machine_get_data
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

トレーニング中にマシンへの接続が失われた可能性があります。データを回復するには、[`wandb sync [PATH_TO_RUN]`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}}) を実行してください。run へのパスは、進行中の Run ID に一致する `wandb` ディレクトリー内のフォルダーです。