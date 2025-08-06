---
title: 1 つのスクリプトから複数の run をローンチするにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

新しい run を開始する前に前回の run を終了させることで、1つのスクリプト内で複数の run をログできます。

推奨される方法は、`wandb.init()` をコンテキストマネージャとして使うことです。これにより、スクリプトで例外が発生した場合は run を終了させて失敗としてマークしてくれます。

```python
import wandb

for x in range(10):
    with wandb.init() as run:
        for y in range(100):
            run.log({"metric": x + y})
```

また、明示的に `run.finish()` を呼び出すこともできます。

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

## 複数のアクティブな run

wandb 0.19.10 以降では、`reinit` 設定を `"create_new"` にすることで、同時に複数のアクティブな run を作成できます。

```python
import wandb

with wandb.init(reinit="create_new") as tracking_run:
    for x in range(10):
        with wandb.init(reinit="create_new") as run:
            for y in range(100):
                run.log({"x_plus_y": x + y})

            tracking_run.log({"x": x})
```

`reinit="create_new"` の詳細や、W&B インテグレーションに関する注意点については、[プロセスごとの複数 run]({{< relref "guides/models/track/runs/multiple-runs-per-process.md" >}}) を参照してください。