---
title: メトリクスだけをログすることはできますか？コードやデータセットの例は必要ありませんか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 管理者
- チーム管理
- メトリクス
---

デフォルトでは、W&B はデータセットの例をログしません。デフォルトで、W&B はコードとシステムメトリクスをログします。

環境変数を使ってコードのログ記録をオフにする方法は 2 つあります：

1. `WANDB_DISABLE_CODE` を `true` に設定すると、すべてのコードトラッキングをオフにできます。この操作により、git SHA や diff パッチの取得が行われなくなります。
2. `WANDB_IGNORE_GLOBS` に `*.patch` を設定すると、diff パッチのサーバーへの同期が停止されますが、`wandb restore` で使用できるようローカルには保持されます。

管理者の場合は、チームの settings からコード保存をオフにすることもできます：

1. `https://wandb.ai/<team>/settings` にアクセスし、チームの settings を開きます。`<team>` にはチーム名が入ります。
2. Privacy セクションまでスクロールします。
3. **Enable code saving by default** を切り替えます。