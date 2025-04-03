---
title: Can I just log metrics, no code or dataset examples?
menu:
  support:
    identifier: ja-support-kb-articles-just_log_metrics_no_code_dataset_examples
support:
- administrator
- team management
- metrics
toc_hide: true
type: docs
url: /support/:filename
---

デフォルトでは、W&B はデータセットの例をログに記録しません。デフォルトでは、W&B はコードとシステムメトリクスをログに記録します。

環境変数を使用してコードのログ記録をオフにするには、次の 2 つのメソッドがあります。

1. `WANDB_DISABLE_CODE` を `true` に設定すると、すべてのコード追跡が無効になります。この操作を行うと、git SHA と差分パッチの取得ができなくなります。
2. `WANDB_IGNORE_GLOBS` を `*.patch` に設定すると、差分パッチの サーバー への同期が停止しますが、`wandb restore` で アプリケーション をローカルで利用できるようになります。

管理者として、 チーム の 設定 で チーム のコード保存をオフにすることもできます。

1. `https://wandb.ai/<team>/settings` で チーム の 設定 に移動します。ここで、`<team>` は チーム の名前です。
2. 「プライバシー」セクションまでスクロールします。
3. **デフォルトでコードの保存を有効にする** を切り替えます。