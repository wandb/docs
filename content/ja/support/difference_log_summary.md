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

表にはサマリーが表示され、ログは将来のプロットのためにすべての値を保存します。

例えば、正確性が変化するたびに `wandb.log` を呼び出します。デフォルトでは、`wandb.log()` は手動でそのメトリックに設定されていない限り、サマリー値を更新します。

散布図と並列座標プロットはサマリー値を使用し、折れ線グラフは `.log` によって記録されたすべての値を表示します。

一部のユーザーは、最新の正確性ではなく最適な正確性を反映させるためにサマリーを手動で設定することを好みます。