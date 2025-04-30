---
title: グリッド検索を再度実行できますか？
menu:
  support:
    identifier: ja-support-kb-articles-rerun_grid_search
support:
- sweep
- ハイパーパラメーター
- runs
toc_hide: true
type: docs
url: /ja/support/:filename
---

クラッシュのために一部の W&B Run の再実行が必要な場合、グリッド検索が完了したにもかかわらず、特定の W&B Run を削除して再実行します。その後、[スイープコントロールページ]({{< relref path="/guides/models/sweeps/sweeps-ui.md" lang="ja" >}})で **再開** ボタンを選択します。新しい Sweep ID を使用して新しい W&B スイープ エージェントを開始します。

完了した W&B Run パラメータの組み合わせは再実行されません。