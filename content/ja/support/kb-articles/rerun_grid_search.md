---
title: グリッド検索を再実行できますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- スイープ
- ハイパーパラメーター
- run
---

グリッド検索が完了したものの、一部の W&B Runs がクラッシュなどで再実行が必要な場合は、該当する W&B Runs を削除して再実行してください。その後、[sweep control page]({{< relref "/guides/models/sweeps/sweeps-ui.md" >}}) で **Resume** ボタンを選択します。新しい Sweep ID を使って新しい W&B Sweep エージェントを起動してください。

すでに完了した W&B Run のパラメータ組み合わせは再実行されません。