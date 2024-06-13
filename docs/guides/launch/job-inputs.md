---
displayed_sidebar: default
---

# ジョブの入力を管理する

Launchのコア体験は、ハイパーパラメーターやデータセットなどの異なるジョブ入力を簡単に実験し、これらのジョブを適切なハードウェアにルーティングすることです。一度ジョブが作成されると、元の作成者以外のユーザーもW&B GUIやCLIを通じてこれらの入力を調整できます。CLIやUIからジョブを起動する際の入力設定方法については、[Enqueue jobs](./add-job-to-queue.md)ガイドを参照してください。

このセクションでは、プログラム的にジョブの入力を制御する方法について説明します。

デフォルトでは、W&Bジョブはジョブの入力として`Run.config`全体をキャプチャしますが、Launch SDKはrun config内の特定のキーを制御したり、JSONやYAMLファイルを入力として指定するための関数を提供します。

:::info
Launch SDKの関数は`wandb-core`を必要とします。詳細は[`wandb-core` README](https://github.com/wandb/wandb/blob/main/core/README.md)を参照してください。
:::

## `Run`オブジェクトの再構成

ジョブ内で`wandb.init`によって返される`Run`オブジェクトは、デフォルトで再構成可能です。Launch SDKは、ジョブを起動する際に再構成可能な`Run.config`オブジェクトの部分をカスタマイズする方法を提供します。

```python
import wandb
from wandb.sdk import launch

# launch sdkの使用に必要
wandb.require("core")

config = {
    "trainer": {
        "learning_rate": 0.01,
        "batch_size": 32,
        "model": "resnet",
        "dataset": "cifar10",
        "private": {
            "key": "value",
        },
    },
    "seed": 42,
}

with wandb.init(config=config)
    launch.manange_wandb_config(
        include=["trainer"], 
        exclude=["trainer.private"],
    )
    # その他
```

`launch.manage_wandb_config`関数は、`Run.config`オブジェクトの入力値を受け入れるようにジョブを設定します。オプションの`include`および`exclude`オプションは、ネストされたconfigオブジェクト内のパスプレフィックスを取ります。例えば、ジョブがユーザーに公開したくないライブラリのオプションを使用する場合に便利です。

`include`プレフィックスが提供されると、config内の`include`プレフィックスに一致するパスのみが入力値を受け入れます。`exclude`プレフィックスが提供されると、`exclude`リストに一致するパスは入力値から除外されます。パスが`include`と`exclude`の両方のプレフィックスに一致する場合、`exclude`プレフィックスが優先されます。

前述の例では、パス`["trainer.private"]`は`trainer`オブジェクトから`private`キーを除外し、パス`["trainer"]`は`trainer`オブジェクト以外のすべてのキーを除外します。

:::tip
名前に`.`が含まれるキーを除外するには、`\`でエスケープされた`.`を使用します。

例えば、`r"trainer\.private"`は`trainer`オブジェクトの`private`キーではなく、`trainer.private`キーを除外します。

上記の`r`プレフィックスは生文字列を示します。
:::

上記のコードがパッケージ化されてジョブとして実行されると、ジョブの入力タイプは次のようになります：

```json
{
    "trainer": {
        "learning_rate": "float",
        "batch_size": "int",
        "model": "str",
        "dataset": "str",
    },
}
```

W&B CLIやUIからジョブを起動する際、ユーザーは4つの`trainer`パラメーターのみを上書きできます。

### run config入力へのアクセス

run config入力を使用して起動されたジョブは、`Run.config`を通じて入力値にアクセスできます。ジョブコード内で`wandb.init`によって返される`Run`は、入力値が自動的に設定されています。ジョブコード内の任意の場所でrun config入力値を読み込むには、次のようにします：
```python
from wandb.sdk import launch

run_config_overrides = launch.load_wandb_config()
```

## ファイルの再構成

Launch SDKは、ジョブコード内のconfigファイルに保存された入力値を管理する方法も提供します。これは、多くのディープラーニングや大規模言語モデルのユースケースで一般的なパターンです。例えば、Hydraを使用したこの[HuggingFace Tune](https://github.com/huggingface/tune/blob/main/configs/benchmark.yaml)の例や、この[Axolotl config](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml)の例などです。

`launch.manage_config_file`関数を使用して、configファイルをLaunchジョブの入力として追加し、ジョブを起動する際にconfigファイル内の値を編集できるようにします。

デフォルトでは、`launch.manage_config_file`が使用されるとrun config入力はキャプチャされません。`launch.manage_wandb_config`を呼び出すと、この振る舞いが上書きされます。

次の例を考えてみましょう：

```python
import yaml
import wandb
from wandb.sdk import launch

# launch sdkの使用に必要
wandb.require("core")

launch.manage_config_file("config.yaml")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

with wandb.init(config=config)
    # その他
```

コードが隣接するファイル`config.yaml`と共に実行されるとします：

```yaml
learning_rate: 0.01
batch_size: 32
model: resnet
dataset: cifar10
```

`launch.manage_config_file`の呼び出しにより、`config.yaml`ファイルがジョブの入力として追加され、W&B CLIやUIから起動する際に再構成可能になります。

`include`および`exclude`キーワード引数は、`launch.manage_wandb_config`と同様にconfigファイルの入力キーをフィルタリングするために使用できます。

### configファイル入力へのアクセス

Launchによって作成されたrunで`launch.manage_config_file`が呼び出されると、`launch`は入力値でconfigファイルの内容をパッチします。パッチされたconfigファイルはジョブ環境で利用可能です。

:::important
ジョブコード内でconfigファイルを読み込む前に`launch.manage_config_file`を呼び出して、入力値が使用されるようにします。
:::