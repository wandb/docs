# スイープ

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_sweep.py#L31-L116)

ハイパーパラメータースイープを初期化します。

```python
sweep(
 sweep: Union[dict, Callable],
 entity: Optional[str] = None,
 project: Optional[str] = None
) -> str
```

スイープからハイパーパラメーターの提案を生成し、それらを使用してモデルをトレーニングするには、このコマンドで返されるsweep_idを使って `wandb.agent` を呼び出します。コマンドライン機能については、コマンドラインツール `wandb sweep` (https://docs.wandb.ai/ref/cli/wandb-sweep)を参照してください。

| 引数 | |
| :--- | :--- |
| `sweep` | 辞書型、SweepConfig型、またはコール可能オブジェクト。スイープ構成（または構成生成器）。dictまたはSweepConfigの場合は、W&Bスイープ構成仕様（https://docs.wandb.ai/guides/sweeps/define-sweep-configuration）に準拠する必要があります。コール可能オブジェクトの場合は、引数なしで呼び出し、W&Bスイープ構成仕様に準拠する辞書を返す必要があります。|
| `entity` | str（オプション）。エンティティは、実行を送信しているユーザー名またはチーム名です。実行を送信する前に、このエンティティが存在している必要があるため、実行をログに記録する前に、アカウントまたはチームをUIで作成してください。エンティティを指定しない場合、実行はデフォルトのエンティティに送信されます。通常、デフォルトのエンティティはユーザー名です。[Settings](https://wandb.ai/settings)の"default location to create new projects"でデフォルトのエンティティを変更してください。|
| `project` | str（オプション）。新しい実行を送信しているプロジェクトの名前。プロジェクトが指定されていない場合、実行は"Uncategorized"プロジェクトに入れられます。|
| 返り値 | |
| :--- | :--- |
| `sweep_id` | str. スイープの一意な識別子。 |



#### 例:

基本的な使い方

```python
import wandb

sweep_configuration = {
 "name": "my-awesome-sweep",
 "metric": {"name": "accuracy", "goal": "maximize"},
 "method": "grid",
 "parameters": {"a": {"values": [1, 2, 3, 4]}},
}


def my_train_func():
 # wandb.configからパラメータ "a" の現在の値を読み取る
 wandb.init()
 a = wandb.config.a

 wandb.log({"a": a, "accuracy": a + 1})
sweep_id = wandb.sweep(sweep_configuration)

# スイープを実行する

wandb.agent(sweep_id, function=my_train_func)

```