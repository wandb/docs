---
title: Can I just log metrics, no code or dataset examples?
menu:
  support:
    identifier: ja-support-just_log_metrics_no_code_dataset_examples
tags:
- administrator
- team management
- metrics
toc_hide: true
type: docs
---

デフォルトでは、W&B はデータセットのサンプルをログに記録しません。デフォルトでは、W&B はコードとシステムメトリクスをログに記録します。

環境変数でコードのログ記録をオフにするには、次の 2 つのメソッドがあります。

1. `WANDB_DISABLE_CODE` を `true` に設定して、すべてのコード追跡をオフにします。このアクションにより、git SHA と差分パッチの取得ができなくなります。
2. `WANDB_IGNORE_GLOBS` を `*.patch` に設定して、差分パッチの サーバー への同期を停止し、`wandb restore` での アプリケーション 用にローカルで利用できるようにします。

管理者として、 チーム の 設定 で チーム のコードの保存をオフにすることもできます。

1. `https://wandb.ai/<team>/settings` で チーム の 設定 に移動します。ここで、`<team>` は チーム の名前です。
2. 「プライバシー」セクションまでスクロールします。
3. **デフォルトでコードの保存を有効にする** を切り替えます。
