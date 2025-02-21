---
title: Best practices to organize hyperparameter searches
menu:
  support:
    identifier: ja-support-best_practices_organize_hyperparameter_searches
tags:
- hyperparameter
- sweeps
- runs
toc_hide: true
type: docs
---

`wandb.init(tags='your_tag')` を使用してユニークなタグを設定します。これにより、プロジェクト ページの Runs テーブルで対応するタグを選択することで、プロジェクト run を効率的にフィルタリングできます。

wandb.init に関する詳細は、[documentation]({{< relref path="/ref/python/init.md" lang="ja" >}}) を参照してください。