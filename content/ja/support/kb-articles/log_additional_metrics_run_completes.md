---
title: How can I log additional metrics after a run completes?
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

実験を管理するには、いくつかの方法があります。

複雑なワークフローでは、複数の run を使用し、[`wandb.init`]({{< relref path="/guides/models/track/launch.md" lang="ja" >}}) の group パラメータを、1つの実験内のすべてのプロセスに対して一意の値に設定します。[**Runs tab**]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ja" >}}) は、テーブルをグループ ID でグループ化し、可視化が適切に機能するようにします。このアプローチにより、1つの場所で結果をログに記録しながら、同時実行の実験とトレーニング run が可能になります。

よりシンプルなワークフローでは、`resume=True` および `id=UNIQUE_ID` を指定して `wandb.init` を呼び出し、同じ `id=UNIQUE_ID` で再度 `wandb.init` を呼び出します。[`wandb.log`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) または `wandb.summary` で通常どおりログを記録すると、run の値がそれに応じて更新されます。