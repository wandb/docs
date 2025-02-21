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

実験管理にはいくつかの方法があります。

複雑なワークフローには、複数の run を使用し、単一の実験内のすべてのプロセスに固有の値を設定するために、[`wandb.init`]({{< relref path="/guides/models/track/launch.md" lang="ja" >}}) でグループパラメータを設定します。[**Runs** タブ]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ja" >}}) はテーブルをグループ ID でまとめ、可視化が正しく機能することを保証します。このアプローチにより、結果を一箇所にログしながら、同時に実験とトレーニング run を行うことができます。

より簡単なワークフローの場合、`wandb.init` を `resume=True` と `id=UNIQUE_ID` で呼び出し、その後同じ `id=UNIQUE_ID` で再度 `wandb.init` を呼び出します。通常の方法で [`wandb.log`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) または `wandb.summary` を使用してログし、run の値はそれに応じて更新されます。