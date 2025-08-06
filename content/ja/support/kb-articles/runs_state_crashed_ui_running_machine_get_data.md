---
title: UI 上では自分の run の状態が `crashed` となっていますが、自分のマシンではまだ動作しています。データを取り戻すにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験管理
---

トレーニング中にマシンとの接続が切断された可能性があります。データを復元するには、[`wandb sync [PATH_TO_RUN]`]({{< relref "/ref/cli/wandb-sync.md" >}}) を実行してください。run へのパスは、実行中の Run ID に一致する `wandb` ディレクトリー内のフォルダーです。