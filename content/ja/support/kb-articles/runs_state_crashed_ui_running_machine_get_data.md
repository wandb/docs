---
title: UI では自分の run の状態が `crashed` となっていますが、実際にはマシン上でまだ動いています。データを復元するにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-runs_state_crashed_ui_running_machine_get_data
support:
- 実験管理
toc_hide: true
type: docs
url: /support/:filename
---

トレーニング中にマシンとの接続が切れた可能性があります。データを復旧するには、[`wandb sync [PATH_TO_RUN]`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}}) を実行してください。run へのパスは、進行中の Run ID と一致する `wandb` ディレクトリー内のフォルダーです。