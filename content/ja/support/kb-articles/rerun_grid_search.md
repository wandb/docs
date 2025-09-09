---
title: グリッド検索を再実行できますか？
menu:
  support:
    identifier: ja-support-kb-articles-rerun_grid_search
support:
- sweeps
- ハイパーパラメーター
- runs
toc_hide: true
type: docs
url: /support/:filename
---

グリッド検索が完了したものの、クラッシュにより再実行が必要な W&B Runs がある場合は、再実行したい該当の W&B Runs を削除してください。その後、[Sweep コントロールページ]({{< relref path="/guides/models/sweeps/sweeps-ui.md" lang="ja" >}}) の **Resume** ボタンを選択してください。新しい Sweep ID を使って新しい W&B sweep agent を起動してください。

完了した W&B Run のパラメータ組み合わせは再実行されません。