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

サマリーはテーブルに表示され、ログは将来のプロットのためにすべての値を保存します。

たとえば、精度が変化するたびに `run.log()` を呼び出します。デフォルトでは、その指標について手動で設定しない限り、`run.log()` はサマリーの値を更新します。

散布図や平行座標プロットはサマリーの値を使用し、折れ線グラフは `run.log` によって記録されたすべての値を表示します。

一部のユーザーは、ログされた直近の精度ではなく最適な精度を反映するように、サマリーを手動で設定することを好みます。