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

`wandb.init(tags='your_tag')` で一意のタグを設定します。これにより、プロジェクトページの Runs Table で対応するタグを選択することにより、プロジェクト の run を効率的にフィルタリングできます。

wandb.int の詳細については、[ドキュメント]({{< relref path="/ref/python/init.md" lang="ja" >}}) を参照してください。
