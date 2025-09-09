---
title: メトリクスだけをログして、コードやデータセットの例なしでも大丈夫ですか？
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

デフォルトでは、W&B はデータセットの例をログしません。デフォルトでは、W&B はコードとシステム メトリクスをログします。

環境変数でコードのログをオフにする方法が 2 つあります:
1. `WANDB_DISABLE_CODE` を `true` に設定して、すべてのコード追跡をオフにします。この操作により、git SHA と diff パッチの取得がされなくなります。
2. `WANDB_IGNORE_GLOBS` を `*.patch` に設定して、diff パッチをサーバーへ同期しないようにします。一方で、`wandb restore` で適用できるようローカルには保持されます。

管理者であれば、チームの設定でコード保存をオフにすることもできます:
1. あなたのチームの設定ページ `https://wandb.ai/<team>/settings` に移動します。`<team>` はあなたのチーム名です。
2. Privacy セクションまでスクロールします。
3. **デフォルトでコード保存を有効にする** を切り替えます。