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

グリッド検索が完了したが、クラッシュのために一部の W&B Run を再実行する必要がある場合、特定の W&B Run を削除して再実行します。その後、[スイープ コントロールページ]({{< relref path="/guides/models/sweeps/sweeps-ui.md" lang="ja" >}})で **Resume** ボタンを選択します。新しい Sweep ID を使用して新しい W&B スイープ エージェントを開始します。

完了した W&B Run パラメータの組み合わせは再実行されません。