---
title: run 完了後に追加のメトリクスをログするにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- run
- メトリクス
---

実験管理にはいくつか方法があります。

複雑なワークフローの場合は、複数の run を使い、[`wandb.init()`]({{< relref "/guides/models/track/launch.md" >}}) 内の group パラメータに単一の実験内ですべてのプロセスに共通のユニークな値を設定してください。[**Runs** タブ]({{< relref "/guides/models/track/project-page.md#runs-tab" >}}) では、group ID ごとにテーブルがグループ化され、可視化が正しく機能します。この方法を使うと、複数の実験やトレーニング run を同時に実行しながら、1 つの場所に結果をログできます。

よりシンプルなワークフローの場合は、`wandb.init()` を `resume=True` と `id=UNIQUE_ID` にして呼び出し、再度同じ `id=UNIQUE_ID` で `wandb.init()` を呼び出してください。[`run.log()`]({{< relref "/guides/models/track/log/" >}}) や `run.summary()` を使って通常通りログを記録すると、その run の値が自動的に最新のものへ更新されます。