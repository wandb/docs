---
title: setup()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-functions-setup
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_setup.py >}}




### <kbd>function</kbd> `setup`

```python
setup(settings: 'Settings | None' = None) → _WandbSetup
```

現在の プロセス と その 子プロセス で W&B を 使えるように 準備します。

通常は `wandb.init()` によって 暗黙に 呼び出されるため、明示的に 呼ぶ必要は ありません。

wandb を 複数の プロセスで 使用する場合、子プロセス を 開始する前に 親プロセス で `wandb.setup()` を 呼び出すと、パフォーマンス や リソース の 利用効率 が 向上することが あります。

`wandb.setup()` は `os.environ` を 変更するため、子プロセス が 変更後の 環境変数 を 継承することが 重要です。

`wandb.teardown()` も 参照してください。



**引数:**
 
 - `settings`:  グローバルに 適用する 設定。これは、その後の `wandb.init()` の 呼び出しで 上書きできます。 



**例:**
 ```python
import multiprocessing

import wandb


def run_experiment(params):
    with wandb.init(config=params):
         # 実験を実行
         pass


if __name__ == "__main__":
    # バックエンドを起動し、グローバルな設定を適用
    wandb.setup(settings={"project": "my_project"})

    # 実験のパラメータを定義
    experiment_params = [
         {"learning_rate": 0.01, "epochs": 10},
         {"learning_rate": 0.001, "epochs": 20},
    ]

    # 複数のプロセスを開始し、各プロセスで個別の実験を実行
    processes = []
    for params in experiment_params:
         p = multiprocessing.Process(target=run_experiment, args=(params,))
         p.start()
         processes.append(p)

    # すべてのプロセスの完了を待機
    for p in processes:
         p.join()

    # 任意: バックエンドを明示的にシャットダウン
    wandb.teardown()
```