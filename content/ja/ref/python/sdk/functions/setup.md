---
title: setup()
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_setup.py >}}




### <kbd>function</kbd> `setup`

```python
setup(settings: 'Settings | None' = None) → _WandbSetup
```

現在のプロセスおよびその子プロセスで W&B を使う準備をします。

通常は、`wandb.init()` が暗黙的にこの関数を呼び出すため、特に意識する必要はありません。

複数のプロセスで wandb を使用する場合、子プロセスを起動する前に親プロセスで `wandb.setup()` を呼ぶことでパフォーマンスやリソースの効率が向上する可能性があります。

`wandb.setup()` は `os.environ` を変更するため、子プロセスが変更された環境変数を継承することが重要です。

関連関数：`wandb.teardown()`


**引数:**
 
 - `settings`:  グローバルに適用される設定。これらは後続の `wandb.init()` で上書きできます。



**例:**
 ```python
import multiprocessing

import wandb


def run_experiment(params):
    with wandb.init(config=params):
         # 実験を実行
         pass


if __name__ == "__main__":
    # バックエンドを開始し、グローバル設定を反映
    wandb.setup(settings={"project": "my_project"})

    # 実験のパラメータを定義
    experiment_params = [
         {"learning_rate": 0.01, "epochs": 10},
         {"learning_rate": 0.001, "epochs": 20},
    ]

    # 複数のプロセスを立ち上げ、それぞれ別の実験を実行
    processes = []
    for params in experiment_params:
         p = multiprocessing.Process(target=run_experiment, args=(params,))
         p.start()
         processes.append(p)

    # 全てのプロセスの完了を待機
    for p in processes:
         p.join()

    # オプション：バックエンドを明示的にシャットダウン
    wandb.teardown()
```