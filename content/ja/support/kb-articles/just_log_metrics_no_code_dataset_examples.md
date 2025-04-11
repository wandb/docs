---
title: メトリクスをログするだけで、コードやデータセットの例は必要ありませんか?
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

デフォルトでは、W&B はデータセットの例をログしません。デフォルトで、W&B はコードとシステムメトリクスをログします。

環境変数でコードログをオフにする方法が2つあります：

1. `WANDB_DISABLE_CODE` を `true` に設定して、すべてのコード追跡をオフにします。この操作により、git SHAと差分パッチの取得が防止されます。
2. `WANDB_IGNORE_GLOBS` を `*.patch` に設定して、アプリケーションを使用して `wandb restore` でローカルに保存し続ける一方で、サーバーへの差分パッチの同期を停止します。

管理者として、チームの設定でコード保存をオフにすることもできます：

1. `https://wandb.ai/<team>/settings` にあるチームの設定に移動します。`<team>` はチームの名前です。
2. プライバシーセクションまでスクロールします。
3. **Enable code saving by default** を切り替えます。