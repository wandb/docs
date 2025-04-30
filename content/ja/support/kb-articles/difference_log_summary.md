---
title: '`.log()` と `.summary` の違いは何ですか？'
menu:
  support:
    identifier: ja-support-kb-articles-difference_log_summary
support:
- チャート
toc_hide: true
type: docs
url: /support/:filename
---

要約は表に表示され、 ログには将来のプロットのためにすべての値が保存されます。

例えば、精度が変わるたびに `wandb.log` を呼び出します。デフォルトでは、そのメトリクスに対するサマリーの値を手動で設定しない限り、`wandb.log()` がサマリー値を更新します。

散布図と並列座標プロットはサマリーの値を使用しますが、ラインプロットは`.log` によって記録されたすべての値を表示します。

一部のユーザーは、 ログに記録された最新の精度ではなく、最適な精度を反映するように、サマリーを手動で設定することを好みます。