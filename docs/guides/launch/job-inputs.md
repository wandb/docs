---
displayed_sidebar: default
---


# Manage job inputs

Launch のコア体験は、ハイパーパラメータやデータセットなどの異なるジョブ入力を簡単に実験し、これらのジョブを適切なハードウェアにルーティングすることです。ジョブが作成されると、元の著者以外のユーザーでも W&B GUI または CLI からこれらの入力を調整できます。CLI または UI からジョブ入力を設定する方法については、[Enqueue jobs](./add-job-to-queue.md) ガイドを参照してください。

このセクションでは、プログラム的にジョブの入力を制御する方法を説明します。

デフォルトでは、W&B ジョブは `Run.config` 全体をジョブの入力としてキャプチャしますが、Launch SDK は run config の選択されたキーを制御したり JSON や YAML ファイルを入力として指定するための関数を提供します。

:::info
Launch SDK の関数は `wandb-core` を必要とします。詳細については [`wandb-core` README](https://github.com/wandb/wandb/blob/main/core/README.md) を参照してください。
:::

## `Run` オブジェクトの再構成

ジョブ内で `wandb.init` によって返される `Run` オブジェクトはデフォルトで再構成可能です。Launch SDK はジョブを起動するときに `Run.config` オブジェクトのどの部分が再構成可能かをカスタマイズする方法を提供します。

```python
import wandb
from wandb.sdk import launch

# Launch SDK の使用に必要。
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
    launch.manage_wandb_config(
        include=["trainer"], 
        exclude=["trainer.private"],
    )
    # その他.
```

関数 `launch.manage_wandb_config` は、ジョブが `Run.config` オブジェクトの入力値を受け入れるように構成します。オプションの `include` および `exclude` オプションはネストされた config オブジェクト内のパスプレフィックスを取ります。これは例えば、ジョブがライブラリを使用し、そのオプションをエンドユーザーに公開したくない場合に役立ちます。

`include` プレフィックスが提供されると、`include` プレフィックスに一致する config 内のパスだけが入力値を受け入れます。`exclude` プレフィックスが提供されると、`exclude` リストに一致するパスは入力値から除外されます。パスが `include` と `exclude` の両方に一致する場合、`exclude` プレフィックスが優先されます。

前述の例では、パス `["trainer.private"]` は `trainer` オブジェクトの `private` キーを除外し、パス `["trainer"]` は `trainer` オブジェクトの下にないすべてのキーを除外します。

:::tip
名前に `.` が含まれるキーをフィルターするには `\` エスケープされた `.` を使用します。

例えば、`r"trainer\.private"` は `trainer.private` キーではなく、`trainer` オブジェクトの下の `private` キーをフィルターします。

上記の `r` プレフィックスは生文字列を示します。
:::

上記のコードがジョブとしてパッケージされて実行されると、ジョブの入力タイプは次のようになります。

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

W&B CLI または UI からジョブを起動すると、ユーザーは 4 つの `trainer` パラメータのみをオーバーライドできます。

### Run config inputs にアクセスする

run config inputs を使用して起動されたジョブは、`Run.config` を通じて入力値にアクセスできます。ジョブコード内で `wandb.init` によって返される `Run` は入力値が自動的に設定されています。ジョブコードのどこでも次のように使用します。

```python
from wandb.sdk import launch

run_config_overrides = launch.load_wandb_config()
```

これにより、run config input values をジョブコードのどこでも読み込むことができます。

## ファイルの再構成

Launch SDK は、ジョブコード内の config ファイルに保存された入力値を管理する方法も提供します。これは多くのディープラーニングおよび大規模言語モデルのユースケースで一般的なパターンです。例として、Hydra を使用したこの [HuggingFace Tune](https://github.com/huggingface/tune/blob/main/configs/benchmark.yaml) またはこの [Axolotl config](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml)) があります。

`launch.manage_config_file` 関数は、config ファイルを Launch ジョブの入力として追加するために使用でき、ジョブの起動時に config ファイル内の値を編集するアクセスを提供します。

デフォルトでは、`launch.manage_config_file` が使用されている場合、run config inputs はキャプチャされません。`launch.manage_wandb_config` を呼び出すことでこの振る舞いを上書きします。

次の例を考えてみてください：

```python
import yaml
import wandb
from wandb.sdk import launch

# Launch SDK の使用に必要。
wandb.require("core")

launch.manage_config_file("config.yaml")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

with wandb.init(config=config)
    # その他.
```

コードが隣接するファイル `config.yaml` とともに実行されると仮定します：

```yaml
learning_rate: 0.01
batch_size: 32
model: resnet
dataset: cifar10
```

`launch.manage_config_file` を呼び出すことで、`config.yaml` ファイルがジョブの入力として追加され、W&B CLI または UI から起動時に再構成可能になります。

キーワード引数 `include` と `exclude` は、`launch.manage_wandb_config` と同じ方法で config ファイルの許容入力キーをフィルターするために使用できます。

### Config file inputs にアクセスする

Launch によって作成された run で `launch.manage_config_file` が呼び出されると、`launch` は入力値で config ファイルの内容をパッチ適用します。パッチ適用された config ファイルはジョブ環境内で利用可能です。

:::important
ジョブコードで config ファイルを読み込む前に `launch.manage_config_file` を呼び出して、入力値が確実に使用されるようにします。
:::