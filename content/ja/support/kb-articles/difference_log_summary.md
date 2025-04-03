---
title: What is the difference between `.log()` and `.summary`?
menu:
  support:
    identifier: ja-support-kb-articles-difference_log_summary
support:
- Charts
toc_hide: true
type: docs
url: /support/:filename
---

サマリーはテーブルに表示され、 ログ には将来のプロットのためにすべての 値 が保存されます。

たとえば、精度が変化するたびに `wandb.log` を呼び出します。デフォルトでは、`wandb.log()` は、そのメトリクスのために手動で設定されない限り、サマリー の 値 を更新します。

散布図と平行座標プロットはサマリー の 値 を使用し、折れ線グラフは `.log` によって記録されたすべての 値 を表示します。

一部の ユーザー は、 ログ に記録された最新の精度ではなく、最適な精度を反映するために、手動でサマリーを設定することを好みます。