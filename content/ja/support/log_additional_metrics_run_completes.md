---
title: How can I log additional metrics after a run completes?
menu:
  support:
    identifier: ja-support-log_additional_metrics_run_completes
tags:
- runs
- metrics
toc_hide: true
type: docs
---

実験 を管理する方法はいくつかあります。

複雑な ワークフロー の場合、複数の run を使用し、[`wandb.init`]({{< relref path="/guides/models/track/launch.md" lang="ja" >}}) で group パラメータ を単一の 実験 内のすべての プロセス に対して一意の 値 に設定します。[**Runs tab**]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ja" >}}) は、テーブルを group ID でグループ化し、 可視化 が適切に機能するようにします。このアプローチにより、並行 実験 および トレーニング run が可能になり、1つの場所に 結果 を ログ 記録できます。

より単純な ワークフロー の場合、`resume=True` と `id=UNIQUE_ID` を指定して `wandb.init` を呼び出し、同じ `id=UNIQUE_ID` で再度 `wandb.init` を呼び出します。[`wandb.log`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) または `wandb.summary` で通常どおり ログ を記録すると、 run の 値 がそれに応じて更新されます。
