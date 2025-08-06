---
title: run 完了後に追加のメトリクスをログするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_additional_metrics_run_completes
support:
- run
- メトリクス
toc_hide: true
type: docs
url: /support/:filename
---

実験管理にはいくつかの方法があります。

複雑なワークフローの場合は、複数の run を使い、[`wandb.init()`]({{< relref path="/guides/models/track/launch.md" lang="ja" >}}) の group パラメータを実験内のすべてのプロセスで共通のユニークな値に設定してください。[**Runs** タブ]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ja" >}}) ではテーブルが group ID でグループ化されるため、可視化が正しく機能します。この方法により、複数の実験やトレーニング run を同時に進めつつ、1 つの場所に結果を記録できます。

シンプルなワークフローの場合は、`wandb.init()` を `resume=True` と `id=UNIQUE_ID` で呼び出し、同じ `id=UNIQUE_ID` でもう一度 `wandb.init()` を実行してください。ログは通常どおり、[`run.log()`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) または `run.summary()` で記録できます。run の値は適宜アップデートされます。