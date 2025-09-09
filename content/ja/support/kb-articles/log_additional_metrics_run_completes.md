---
title: run が完了した後に追加のメトリクスをログするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_additional_metrics_run_completes
support:
- runs
- メトリクス
toc_hide: true
type: docs
url: /support/:filename
---

実験を管理する方法はいくつかあります。

複雑なワークフローでは、複数の run を使い、[`wandb.init()`]({{< relref path="/guides/models/track/create-an-experiment.md" lang="ja" >}}) の group パラメータ を単一の実験内のすべてのプロセスで同じ一意の 値 に設定します。[**Runs** タブ]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ja" >}}) はテーブルを group ID ごとにグループ化し、可視化が正しく機能するようにします。この方法なら、結果を 1 か所にログしながら、実験とトレーニング run を同時に実行できます。

より単純なワークフローでは、`resume=True` と `id=UNIQUE_ID` を指定して `wandb.init()` を呼び、同じ `id=UNIQUE_ID` で再度 `wandb.init()` を呼びます。ログは [`run.log()`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) や `run.summary()` で通常どおり行えば、run の 値 が適切に更新されます。