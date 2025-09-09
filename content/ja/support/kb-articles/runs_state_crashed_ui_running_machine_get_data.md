---
title: 私の run の状態が UI 上では `crashed` になっていますが、ローカルのマシンではまだ実行中です。データを取り戻すにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-runs_state_crashed_ui_running_machine_get_data
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

トレーニング中にマシンとの接続が切れた可能性があります。[`wandb sync [PATH_TO_RUN]`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}})を実行してデータを復旧してください。あなたの run へのパスは、`wandb` ディレクトリー内の、実行中の run の Run ID に一致するフォルダーです。