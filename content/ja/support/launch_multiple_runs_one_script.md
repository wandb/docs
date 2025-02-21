---
title: How do I launch multiple runs from one script?
menu:
  support:
    identifier: ja-support-launch_multiple_runs_one_script
tags:
- experiments
toc_hide: true
type: docs
---

`wandb.init` と `run.finish()` を使用して、1つの スクリプト 内で複数の run を ログ 記録します。

1. `run = wandb.init(reinit=True)` を使用して、run の再初期化を許可します。
2. 各 run の最後に `run.finish()` を呼び出して、ログ 記録を完了させます。

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    for y in range(100):
        wandb.log({"metric": x + y})
    run.finish()
```

あるいは、Python コンテキストマネージャーを利用して、ログ 記録を自動的に終了します。

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    with run:
        for y in range(100):
            run.log({"metric": x + y})
```