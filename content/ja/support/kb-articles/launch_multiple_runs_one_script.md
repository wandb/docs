---
title: 1 つのスクリプトから複数の run をローンンチするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-launch_multiple_runs_one_script
support:
- 実験管理
toc_hide: true
type: docs
url: /support/:filename
---

新しい run を開始する前に前の run を終了して、1 つの スクリプト 内で複数の runs を ログ してください。

これを行う推奨方法は、`wandb.init()` を コンテキストマネージャー として使うことです。こうすると、スクリプトが例外を送出した場合にその run を終了し、失敗としてマークします:

```python
import wandb

for x in range(10):
    with wandb.init() as run:
        for y in range(100):
            run.log({"metric": x + y})
```

`run.finish()` を明示的に呼ぶこともできます:

```python
import wandb

for x in range(10):
    run = wandb.init()

    try:
        for y in range(100):
            run.log({"metric": x + y})

    except Exception:
        run.finish(exit_code=1)
        raise

    finally:
        run.finish()
```

## 複数のアクティブな runs

wandb 0.19.10 以降では、同時にアクティブな複数の runs を作成するために、`reinit` 設定を `"create_new"` に設定できます。

```python
import wandb

with wandb.init(reinit="create_new") as tracking_run:
    for x in range(10):
        with wandb.init(reinit="create_new") as run:
            for y in range(100):
                run.log({"x_plus_y": x + y})

            tracking_run.log({"x": x})
```

`reinit="create_new"` に関する詳細 (W&B インテグレーション における注意点を含む) については、[プロセス ごとの複数の runs]({{< relref path="guides/models/track/runs/multiple-runs-per-process.md" lang="ja" >}}) を参照してください。