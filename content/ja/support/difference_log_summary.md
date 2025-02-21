---
title: What is the difference between `.log()` and `.summary`?
menu:
  support:
    identifier: ja-support-difference_log_summary
tags:
- Charts
toc_hide: true
type: docs
---

サマリーはテーブルに表示され、 ログ には将来のプロットのためにすべての 値 が保存されます。

たとえば、精度が変化するたびに `wandb.log` を呼び出します。デフォルトでは、`wandb.log()` は、そのメトリクスに対して手動で設定されていない限り、サマリー の 値 を更新します。

散布図と平行座標プロットはサマリー の 値 を使用し、折れ線グラフは `.log` によって記録されたすべての 値 を表示します。

一部の ユーザー は、 ログ に記録された最新の精度ではなく、最適な精度を反映するように、サマリーを手動で設定することを好みます。
