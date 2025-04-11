---
title: run が完了した後に追加のメトリクスをログするにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_additional_metrics_run_completes
support:
- runs
- metrics
toc_hide: true
type: docs
url: /support/:filename
---

実験管理にはいくつかの方法があります。

複雑なワークフローの場合、複数の run を使用し、[`wandb.init`]({{< relref path="/guides/models/track/launch.md" lang="ja" >}}) でグループパラメータを設定し、すべてのプロセスに対して一意の値を持たせます。これにより、[**Runs** タブ]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ja" >}}) はテーブルをグループ ID でまとめ、可視化を正しく機能させます。このアプローチでは、結果を1か所にログしながら、同時に実験とトレーニング run を行うことができます。

よりシンプルなワークフローの場合、`wandb.init` を `resume=True` および `id=UNIQUE_ID` とともに呼び出し、同じ `id=UNIQUE_ID` でもう一度 `wandb.init` を呼び出します。通常通り [`wandb.log`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) または `wandb.summary` でログを記録し、run の値はそれに応じて更新されます。