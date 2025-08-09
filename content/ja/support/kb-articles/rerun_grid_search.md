---
title: グリッド検索を再実行できますか？
menu:
  support:
    identifier: ja-support-kb-articles-rerun_grid_search
support:
- スイープ
- ハイパーパラメーター
- runs
toc_hide: true
type: docs
url: /support/:filename
---

グリッド検索が完了していても、一部の W&B Run がクラッシュなどで再実行が必要な場合、再実行したい特定の W&B Run を削除してください。その後、[sweep コントロールページ]({{< relref path="/guides/models/sweeps/sweeps-ui.md" lang="ja" >}})で **Resume** ボタンを選択します。新しい Sweep ID を使って新たに W&B Sweep agent を起動してください。

すでに完了している W&B Run のパラメータの組み合わせは再実行されません。