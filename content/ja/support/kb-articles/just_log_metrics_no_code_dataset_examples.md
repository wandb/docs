---
title: メトリクスだけをログすることはできますか？コードやデータセットの例は必要ありませんか？
menu:
  support:
    identifier: ja-support-kb-articles-just_log_metrics_no_code_dataset_examples
support:
- 管理者
- チーム管理
- メトリクス
toc_hide: true
type: docs
url: /support/:filename
---

デフォルトでは、W&B はデータセットの例をログしません。W&B はデフォルトでコードとシステムメトリクスをログします。

環境変数を使ってコードのロギングをオフにする方法は 2 つあります。

1. `WANDB_DISABLE_CODE` を `true` に設定すると、すべてのコードトラッキングが無効になります。この設定により、git SHA や diff パッチの取得が行われなくなります。
2. `WANDB_IGNORE_GLOBS` を `*.patch` に設定すると、diff パッチをサーバーに同期せず、ローカルにのみ保存されます（`wandb restore` で適用可能）。

管理者の場合は、チームの設定からチーム全体のコード保存をオフにすることもできます。

1. `https://wandb.ai/<team>/settings` にアクセスしてください。`<team>` にはご自身のチーム名を入力します。
2. 「プライバシー」セクションまでスクロールします。
3. **Enable code saving by default** の切り替えスイッチを操作します。