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

サマリーはテーブルに表示され、ログにはすべての値が保存されて将来のプロットに使えます。

例えば、精度が変化するたびに `run.log()` を呼び出します。デフォルトでは、`run.log()` はそのメトリクスに対するサマリー値を自動的に更新しますが、手動で設定することも可能です。

散布図やパラレル座標プロットにはサマリー値が使われ、折れ線グラフには `run.log` で記録されたすべての値が表示されます。

最も最近ログした精度ではなく、最適な精度を反映するためにサマリーを手動で設定するユーザーもいます。