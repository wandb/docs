# エージェント

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/wandb_agent.py#L592-L650)

サーバーによって指定された設定パラメーターを使用して、関数やプログラムを実行します。
```python
sweep_id, function = None, entity = None, project = None, count = None
```



| :--- | :--- |
| `function` | (func, 任意) 設定で指定された"program"の代わりに呼び出す関数。|
| `project` | (str, 任意) W＆Bプロジェクト |
#### 例:

関数上でサンプルスイープを実行する:

```python
import wandb

sweep_configuration = {
    "name": "my-awesome-sweep",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "method": "grid",
    "parameters": {"a": {"values": [1, 2, 3, 4]}},
}


def my_train_func():
    # wandb.config からパラメータ "a" の現在の値を読み取る
    wandb.init()
    a = wandb.config.a

    wandb.log({"a": a, "accuracy": a + 1})


sweep_id = wandb.sweep(sweep_configuration)

# スイープを実行する
wandb.agent(sweep_id, function=my_train_func)
```