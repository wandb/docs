---
title: ハイパーパラメーター探索を整理するためのベストプラクティス
menu:
  support:
    identifier: ja-support-kb-articles-best_practices_organize_hyperparameter_searches
support:
- ハイパーパラメーター
- スイープ
- run
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init(tags='your_tag')` でユニークなタグを設定できます。これにより、 Project ページの Runs テーブルで該当するタグを選択することで、プロジェクトの run を効率的にフィルタリングできます。

`wandb.init()` の詳細については、[`wandb.init()` のリファレンス]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}})をご覧ください。