---
title: 人間が読みやすい Run 名をプログラムで取得するにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-programmatically_access_humanreadable_run_name
support:
  - experiments
toc_hide: true
type: docs
url: /ja/support/:filename
translationKey: programmatically_access_humanreadable_run_name
---
`.name` 属性は、[`wandb.Run`]({{< relref path="/ref/python/run.md" lang="ja" >}}) から以下のようにアクセスできます:

```python
import wandb

wandb.init()
run_name = wandb.run.name
```