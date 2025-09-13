---
title: ハイパラメーター探索を整理するためのベストプラクティス
menu:
  support:
    identifier: ja-support-kb-articles-best_practices_organize_hyperparameter_searches
support:
- ハイパーパラメーター
- sweeps
- runs
toc_hide: true
type: docs
url: /support/:filename
---

一意のタグは `wandb.init(tags='your_tag')` で設定できます。これにより、Project ページの Runs Table で該当するタグを選択して、Project の Runs を効率的にフィルタリングできます。

`wandb.init()` の詳細は、[`wandb.init()` リファレンス]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) を参照してください。