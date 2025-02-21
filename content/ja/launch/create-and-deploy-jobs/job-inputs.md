---
title: Manage job inputs
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-job-inputs
    parent: create-and-deploy-jobs
url: guides/launch/job-inputs
---

Launch の中心的な体験は、ハイパーパラメーターやデータセットのような異なるジョブ入力を簡単に試し、これらのジョブを適切なハードウェアにルーティングすることです。ジョブが作成されると、元の作成者以外の ユーザー も W&B GUI または CLI を介してこれらの入力を調整できます。CLI または UI から起動する際にジョブ入力を設定する方法については、[ジョブのエンキュー]({{< relref path="./add-job-to-queue.md" lang="ja" >}}) ガイドを参照してください。

このセクションでは、ジョブで調整できる入力をプログラムで制御する方法について説明します。

デフォルトでは、W&B ジョブは `Run.config` 全体をジョブへの入力としてキャプチャしますが、Launch SDK は run config 内の選択した キー を制御したり、JSON または YAML ファイルを入力として指定したりする機能を提供します。

{{% alert %}}
Launch SDK の関数には `wandb-core` が必要です。詳細については、[`wandb-core` の README](https://github.com/wandb/wandb/blob/main/core/README.md) を参照してください。
{{% /alert %}}

## `Run` オブジェクトの再設定

ジョブ内の `wandb.init` によって返される `Run` オブジェクトは、デフォルトで再設定できます。Launch SDK は、ジョブの起動時に `Run.config` オブジェクトのどの部分を再設定できるかをカスタマイズする方法を提供します。

```python
import wandb
from wandb.sdk import launch

# Required for launch sdk use.
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


with wandb.init(config=config):
    launch.manage_wandb_config(
        include=["trainer"], 
        exclude=["trainer.private"],
    )
    # Etc.
```

関数 `launch.manage_wandb_config` は、`Run.config` オブジェクトの入力値を 受け入れるようにジョブを構成します。オプションの `include` および `exclude` オプションは、ネストされた config オブジェクト内のパスプレフィックスを取ります。これは、たとえば、ジョブがエンド ユーザー に公開したくないオプションを持つ ライブラリ を使用する場合に役立ちます。

`include` プレフィックスが指定されている場合、`include` プレフィックスに一致する config 内のパスのみが入力値を 受け入れます。`exclude` プレフィックスが指定されている場合、`exclude` リストに一致するパスは入力値から除外されません。パスが `include` と `exclude` の両方のプレフィックスに一致する場合、`exclude` プレフィックスが優先されます。

上記の例では、パス `["trainer.private"]` は `private` キー を `trainer` オブジェクトから除外し、パス `["trainer"]` は `trainer` オブジェクトにないすべての キー を除外します。

{{% alert %}}
名前 に `.` がある キー を除外するには、`\` でエスケープされた `.` を使用します。

たとえば、`r"trainer\.private"` は、`trainer` オブジェクトの `private` キー ではなく、`trainer.private` キー を除外します。

上記の `r` プレフィックスは、raw 文字列を表すことに注意してください。
{{% /alert %}}

上記の コード がパッケージ化され、ジョブとして実行される場合、ジョブの入力タイプは次のようになります。

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

W&B CLI または UI からジョブを起動すると、 ユーザー は4つの `trainer` パラメータ のみをオーバーライドできます。

### run config 入力へのアクセス

run config 入力で起動されたジョブは、`Run.config` を介して入力値にアクセスできます。ジョブ コード の `wandb.init` によって返される `Run` には、入力値が自動的に設定されます。ジョブ コード の任意の場所で run config 入力値をロードするには、
```python
from wandb.sdk import launch

run_config_overrides = launch.load_wandb_config()
```
を使用します。

## ファイルの再構成

Launch SDK は、ジョブ コード の config ファイルに保存されている入力値を管理する方法も提供します。これは、この [torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_lora.yaml) の例やこの [Axolotl config](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml)) のように、多くの ディープラーニング および大規模言語モデルの ユースケース で一般的なパターンです。

{{% alert %}}
[Launch での Sweeps]({{< relref path="../sweeps-on-launch.md" lang="ja" >}}) は、sweep パラメータ として config ファイル入力の使用をサポートしていません。Sweep パラメータ は `Run.config` オブジェクトを介して制御する必要があります。
{{% /alert %}}

`launch.manage_config_file` 関数を使用すると、config ファイルを Launch ジョブへの入力として追加でき、ジョブの起動時に config ファイル内の値を編集する アクセス 権を付与できます。

デフォルトでは、`launch.manage_config_file` が使用されている場合、run config 入力はキャプチャされません。`launch.manage_wandb_config` を呼び出すと、この 振る舞い がオーバーライドされます。

次の例を考えてみましょう。

```python
import yaml
import wandb
from wandb.sdk import launch

# Required for launch sdk use.
wandb.require("core")

launch.manage_config_file("config.yaml")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

with wandb.init(config=config):
    # Etc.
    pass
```

コード が隣接するファイル `config.yaml` で実行されると想像してください。

```yaml
learning_rate: 0.01
batch_size: 32
model: resnet
dataset: cifar10
```

`launch.manage_config_file` を呼び出すと、`config.yaml` ファイルがジョブへの入力として追加され、W&B CLI または UI から起動するときに再構成可能になります。

`include` および `exclude` キーワード arugment は、`launch.manage_wandb_config` と同じ方法で、config ファイルの許容される入力 キー をフィルタリングするために使用できます。

### config ファイル入力へのアクセス

Launch によって作成された run で `launch.manage_config_file` が呼び出されると、`launch` は config ファイルの内容を入力値でパッチします。パッチされた config ファイルは、ジョブ 環境 で使用できます。

{{% alert color="secondary" %}}
入力値が使用されるようにするには、ジョブ コード で config ファイルを読み取る前に `launch.manage_config_file` を呼び出してください。
{{% /alert %}}

### ジョブの Launch ドロワー UI のカスタマイズ

ジョブの入力の スキーマ を定義すると、ジョブを起動するためのカスタム UI を作成できます。ジョブの スキーマ を定義するには、`launch.manage_wandb_config` または `launch.manage_config_file` の呼び出しに含めます。スキーマ は、[JSON Schema](https://json-schema.org/understanding-json-schema/reference) の形式の python 辞書 、または Pydantic モデル クラスのいずれかになります。

{{% alert color="secondary" %}}
ジョブ入力 スキーマ は、入力を検証するために使用されません。これらは、Launch ドロワーの UI を定義するためにのみ使用されます。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "JSON schema" %}}
次の例は、次のプロパティを持つ スキーマ を示しています。

- `seed` 、整数
- `trainer` 、いくつかの キー が指定された 辞書 :
  - `trainer.learning_rate` 、ゼロより大きい必要がある float
  - `trainer.batch_size` 、16、64、または 256 のいずれかである必要がある整数
  - `trainer.dataset` 、 `cifar10` または `cifar100` のいずれかである必要がある文字列

```python
schema = {
    "type": "object",
    "properties": {
        "seed": {
          "type": "integer"
        }
        "trainer": {
            "type": "object",
            "properties": {
                "learning_rate": {
                    "type": "number",
                    "description": "Learning rate of the model",
                    "exclusiveMinimum": 0,
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Number of samples per batch",
                    "enum": [16, 64, 256]
                },
                "dataset": {
                    "type": "string",
                    "description": "Name of the dataset to use",
                    "enum": ["cifar10", "cifar100"]
                }
            }
        }
    }
}

launch.manage_wandb_config(
    include=["seed", "trainer"], 
    exclude=["trainer.private"],
    schema=schema,
)
```

一般に、次の JSON Schema 属性がサポートされています。

| 属性 | 必須 | 注 |
| --- | --- | --- |
| `type` | はい | `number` 、 `integer` 、 `string` 、または `object` のいずれかである必要があります |
| `title` | いいえ | プロパティ の表示名をオーバーライドします |
| `description` | いいえ | プロパティ ヘルパーテキストを提供します |
| `enum` | いいえ | フリーフォームテキストエントリの代わりにドロップダウン選択を作成します |
| `minimum` | いいえ | `type` が `number` または `integer` の場合にのみ許可されます |
| `maximum` | いいえ | `type` が `number` または `integer` の場合にのみ許可されます |
| `exclusiveMinimum` | いいえ | `type` が `number` または `integer` の場合にのみ許可されます |
| `exclusiveMaximum` | いいえ | `type` が `number` または `integer` の場合にのみ許可されます |
| `properties` | いいえ | `type` が `object` の場合、ネストされた 構成 を定義するために使用されます |
{{% /tab %}}
{{% tab "Pydantic model" %}}
次の例は、次のプロパティを持つ スキーマ を示しています。

- `seed` 、整数
- `trainer` 、いくつかのサブ属性が指定された スキーマ :
  - `trainer.learning_rate` 、ゼロより大きい必要がある float
  - `trainer.batch_size` 、1 以上 256 以下の整数
  - `trainer.dataset` 、 `cifar10` または `cifar100` のいずれかである必要がある文字列

```python
class DatasetEnum(str, Enum):
    cifar10 = "cifar10"
    cifar100 = "cifar100"

class Trainer(BaseModel):
    learning_rate: float = Field(gt=0, description="Learning rate of the model")
    batch_size: int = Field(ge=1, le=256, description="Number of samples per batch")
    dataset: DatasetEnum = Field(title="Dataset", description="Name of the dataset to use")

class Schema(BaseModel):
    seed: int
    trainer: Trainer

launch.manage_wandb_config(
    include=["seed", "trainer"],
    exclude=["trainer.private"],
    schema=Schema,
)
```

クラス のインスタンスを使用することもできます。

```python
t = Trainer(learning_rate=0.01, batch_size=32, dataset=DatasetEnum.cifar10)
s = Schema(seed=42, trainer=t)
launch.manage_wandb_config(
    include=["seed", "trainer"],
    exclude=["trainer.private"],
    input_schema=s,
)
```
{{% /tab %}}
{{< /tabpane >}}

ジョブ入力 スキーマ を追加すると、Launch ドロワーに構造化されたフォームが作成され、ジョブの起動が容易になります。

{{< img src="/images/launch/schema_overrides.png" alt="" >}}
