---
title: My run's state is `crashed` on the UI but is still running on my machine. What
  do I do to get my data back?
menu:
  support:
    identifier: ja-support-kb-articles-runs_state_crashed_ui_running_machine_get_data
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

トレーニング 中 に マシン と の 接続 が 失わ れ た 可能 性 が あり ます。[`wandb sync [PATH_TO_RUN]`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}}) を 実行 し て データ を 復元 し ます。run へ の パス は、実行 中 の Run ID と 一致 する `wandb` ディレクトリー 内 の フォルダー です。