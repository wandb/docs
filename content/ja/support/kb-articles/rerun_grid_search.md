---
title: Can I rerun a grid search?
menu:
  support:
    identifier: ja-support-kb-articles-rerun_grid_search
support:
- sweeps
- hyperparameter
- runs
toc_hide: true
type: docs
url: /support/:filename
---

グリッド検索 が完了したが、クラッシュが原因で一部の W&B Runs の再実行が必要な場合は、特定の W&B Runs を削除して再実行します。次に、[sweep control page]({{< relref path="/guides/models/sweeps/sweeps-ui.md" lang="ja" >}}) の [**Resume**] ボタンを選択します。新しい Sweep ID を使用して、新しい W&B Sweep agent を開始します。

完了した W&B Run のパラメータの組み合わせは、再実行されません。
