---
title: 1 つのスクリプトから複数の run をどうやってローンチしますか？
menu:
  support:
    identifier: ja-support-kb-articles-launch_multiple_runs_one_script
support:
- 実験管理
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init` と `run.finish()` を使用して、単一のスクリプト内で複数の run をログする方法:

1. `run = wandb.init(reinit=True)` を使用して、run の再初期化を許可します。
2. 各 run の最後に `run.finish()` を呼び出して、ログを完了します。

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    for y in range(100):
        wandb.log({"metric": x + y})
    run.finish()
```

または、Python のコンテキストマネージャを利用して自動的にログを完了します:

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    with run:
        for y in range(100):
            run.log({"metric": x + y})
```
