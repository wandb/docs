---
title: 1 つのスクリプトから複数の run をローンチするにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-launch_multiple_runs_one_script
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

前の run を終了してから新しい run を開始すると、1 つのスクリプト内で複数の run をログできます。

これを行う推奨方法は、`wandb.init()` をコンテキストマネージャーとして使うことです。こうすることで、スクリプトが例外を投げた場合でも run を終了し、失敗としてマークします。

```python
import wandb

for x in range(10):
    # run をコンテキストマネージャーとして開始
    with wandb.init() as run:
        for y in range(100):
            # メトリクスをログ
            run.log({"metric": x + y})
```

`run.finish()` を明示的に呼び出すこともできます。

```python
import wandb

for x in range(10):
    run = wandb.init()

    try:
        for y in range(100):
            # メトリクスをログ
            run.log({"metric": x + y})

    except Exception:
        # 例外発生時に run を失敗として終了
        run.finish(exit_code=1)
        raise

    finally:
        # 最後に必ず run を終了
        run.finish()
```

## 複数のアクティブ run

wandb 0.19.10 以降では、`reinit` 設定に `"create_new"` をセットすることで、同時に複数のアクティブな run を作成できます。

```python
import wandb

with wandb.init(reinit="create_new") as tracking_run:
    for x in range(10):
        with wandb.init(reinit="create_new") as run:
            for y in range(100):
                # x + y の値をログ
                run.log({"x_plus_y": x + y})

            # 各ループごとに x の値を tracking_run へログ
            tracking_run.log({"x": x})
```

`reinit="create_new"` についての詳細や W&B インテグレーションでの注意点は、[プロセスごとの複数 run]({{< relref path="guides/models/track/runs/multiple-runs-per-process.md" lang="ja" >}}) を参照してください。