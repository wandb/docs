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

`wandb.init` と `run.finish()` を使用して、単一のスクリプト内で複数の run をログ:

1. `run = wandb.init(reinit=True)` を使用して、run の再初期化を許可する。
2. 各 run の最後で `run.finish()` を呼び出してログを完了させる。

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    for y in range(100):
        wandb.log({"metric": x + y})
    run.finish()
```

または、Python のコンテキストマネージャを利用して、自動的にログを完了させる:

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    with run:
        for y in range(100):
            run.log({"metric": x + y})
```