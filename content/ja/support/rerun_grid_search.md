---
title: Can I rerun a grid search?
menu:
  support:
    identifier: ja-support-rerun_grid_search
tags:
- sweeps
- hyperparameter
- runs
toc_hide: true
type: docs
---

グリッド検索 が完了したが、クラッシュが原因でいくつかの W&B Runs の再実行が必要な場合は、特定の W&B Runs を削除して再実行してください。次に、[スイープコントロールページ]({{< relref path="/guides/models/sweeps/sweeps-ui.md" lang="ja" >}}) で [**再開**] ボタンを選択します。新しい Sweep ID を使用して、新しい W&B Sweep agent を開始します。

完了した W&B Run パラメータの組み合わせは再実行されません。
