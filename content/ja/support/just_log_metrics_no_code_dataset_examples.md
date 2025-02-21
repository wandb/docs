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

W&B はデフォルトでデータセットの例をログしません。デフォルトでは、W&B はコードとシステムメトリクスをログします。

環境変数を使用してコードのログをオフにする方法が 2 つあります:

1. `WANDB_DISABLE_CODE` を `true` に設定して、すべてのコードトラッキングをオフにします。このアクションは、git SHA と diff パッチの取得を防ぎます。
2. `WANDB_IGNORE_GLOBS` を `*.patch` に設定することで、サーバーへの diff パッチの同期を停止し、`wandb restore` を使用してアプリケーションにローカルで利用可能な状態を維持します。

管理者として、チームの設定でチームのためのコード保存をオフにすることもできます:

1. `https://wandb.ai/<team>/settings` で、チームの設定に移動します。`<team>` はチームの名前です。
2. プライバシーセクションまでスクロールします。
3. **Enable code saving by default** を切り替えます。