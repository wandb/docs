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

現在のプロセスおよびその子プロセスで W&B を使うための準備を行います。

通常は `wandb.init()` が内部で自動的に呼ばれるため、明示的に呼び出す必要はありません。

複数のプロセスで wandb を利用する場合、親プロセスで `wandb.setup()` を最初に呼び出してから子プロセスを起動することで、パフォーマンスやリソース効率が向上することがあります。

`wandb.setup()` は `os.environ` を変更するため、子プロセスがその変更後の環境変数を継承することが重要です。

`wandb.teardown()` もあわせてご参照ください。



**引数:**
 
 - `settings`:  グローバルに適用される設定。これらは、後の `wandb.init()` の呼び出しによって上書きされることがあります。



**使用例:**
 ```python
import multiprocessing

import wandb


def run_experiment(params):
    with wandb.init(config=params):
         # 実験を実行
         pass


if __name__ == "__main__":
    # バックエンドを起動し、グローバル設定を指定
    wandb.setup(settings={"project": "my_project"})

    # 実験のパラメータを定義
    experiment_params = [
         {"learning_rate": 0.01, "epochs": 10},
         {"learning_rate": 0.001, "epochs": 20},
    ]

    # 複数のプロセスを立ち上げ、それぞれで別個の実験を実行
    processes = []
    for params in experiment_params:
         p = multiprocessing.Process(target=run_experiment, args=(params,))
         p.start()
         processes.append(p)

    # すべてのプロセスが完了するまで待機
    for p in processes:
         p.join()

    # オプション: バックエンドのシャットダウンを明示的に実行
    wandb.teardown()
```