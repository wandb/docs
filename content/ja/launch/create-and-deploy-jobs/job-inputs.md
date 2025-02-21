---
title: Manage job inputs
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-job-inputs
    parent: create-and-deploy-jobs
url: guides/launch/job-inputs
---

Launch のコア経験は、ハイパーパラメーターやデータセットなどの異なるジョブ入力を簡単に実験し、これらのジョブを適切なハードウェアにルーティングすることです。一度ジョブが作成されると、元の作者以外のユーザーも W&B の GUI または CLI 経由でこれらの入力を調整することができます。CLI または UI からのローンンチ時にジョブ入力を設定する方法については、[Enqueue jobs]({{< relref path="./add-job-to-queue.md" lang="ja" >}}) ガイドをご覧ください。

このセクションでは、プログラムでジョブのために調整可能な入力を制御する方法を説明します。

デフォルトでは、W&B ジョブは `Run.config` 全体をジョブの入力としてキャプチャしますが、Launch SDK は run config の中の選択したキーを制御したり、入力として JSON または YAML ファイルを指定するための機能を提供します。

{{% alert %}}
Launch SDK の機能は `wandb-core` を必要とします。詳細は [`wandb-core` README](https://github.com/wandb/wandb/blob/main/core/README.md) をご覧ください。
{{% /alert %}}

## `Run` オブジェクトの再設定

ジョブで `wandb.init` によって返される `Run` オブジェクトは、デフォルトで再設定可能です。Launch SDK は、ジョブをローンンチする際に再設定可能な `Run.config` オブジェクトのどの部分をカスタマイズできるかを指定する方法を提供します。

```python
import wandb
from wandb.sdk import launch

# Launch SDK の使用に必要です。
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
    # その他のコード
```

`launch.manage_wandb_config` 関数は `Run.config` オブジェクトの入力値を受け入れるようにジョブを構成します。オプションの `include` および `exclude` オプションは、ネストされた設定オブジェクト内のパスプレフィックスを取ります。例えば、ジョブがライブラリを利用しており、そのオプションをエンドユーザーに公開したくない場合に活用できます。

`include` プレフィックスが指定されている場合、config 内で `include` プレフィックスに一致するパスのみが入力値を受け入れます。`exclude` プレフィックスが指定されている場合、`exclude` リストに一致するパスは入力値から除外されます。パスが `include` と `exclude` の両方のプレフィックスに一致する場合、`exclude` プレフィックスが優先されます。

上記の例では、パス `["trainer.private"]` は `trainer` オブジェクトの `private` キーをフィルタリングし、パス `["trainer"]` は `trainer` オブジェクトの下にないすべてのキーをフィルタリングします。

{{% alert %}}
名前に `.` を含むキーをフィルタリングするには、`\` でエスケープされた `.` を使用してください。

例えば、`r"trainer\.private"` は `trainer.private` キーではなく、`trainer` オブジェクトの下の `private` キーをフィルタリングします。

上記の `r` プレフィックスは生文字列を示します。
{{% /alert %}}

上記のコードがパッケージ化され、ジョブとして実行される場合、ジョブの入力タイプは次のようになります:

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

W&B CLI または UI からジョブをローンンチする際、ユーザーは4つの `trainer` パラメーターのみを上書きすることができます。

### Run config inputs へのアクセス

run config inputs と共にローンンチされたジョブは、`Run.config` を通じて入力値にアアクセスできます。ジョブコード内で `wandb.init` によって返される `Run` は、入力値が自動的に設定されます。ジョブコード内で任意の場所で run config 入力値をロードするために次を使用します。

```python
from wandb.sdk import launch

run_config_overrides = launch.load_wandb_config()
```

## ファイルの再設定

Launch SDK は、ジョブコード内の設定ファイルに格納された入力値を管理する方法も提供します。これは、多くのディープラーニングや大規模言語モデルユースケースで一般的なパターンです。この [torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_lora.yaml) の例やこの [Axolotl config](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml) の例のように。

{{% alert %}}
[Launch の Sweeps]({{< relref path="../sweeps-on-launch.md" lang="ja" >}}) は設定ファイル入力をスイープパラメータとして使用することをサポートしていません。スイープパラメータは `Run.config` オブジェクトを通じて制御する必要があります。
{{% /alert %}}

`launch.manage_config_file` 関数を使用すると、config ファイルを Launch ジョブへの入力として追加し、ジョブのローンンチ時に config ファイル内の値を編集することができます。

デフォルトでは、`launch.manage_config_file` が使用される場合、run config の入力はキャプチャされません。`launch.manage_wandb_config` を呼び出すと、この振る舞いがオーバーライドされます。

次の例を考えてみてください:

```python
import yaml
import wandb
from wandb.sdk import launch

# Launch SDK の使用に必要です。
wandb.require("core")

launch.manage_config_file("config.yaml")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

with wandb.init(config=config):
    # その他のコード
    pass
```

コードが近接したファイル `config.yaml` とともに実行されることを想像してみてください:

```yaml
learning_rate: 0.01
batch_size: 32
model: resnet
dataset: cifar10
```

`launch.manage_config_file` の呼び出しは `config.yaml` ファイルをジョブの入力として追加し、W&B CLI または UI からローンンチする際に再設定可能にします。

`launch.manage_wandb_config` と同様に `include` および `exclude` キーワード引数を使用して、config ファイルの許可される入力キーをフィルタリングできます。

### Config file inputs へのアクセス

Launch によって作成された run で `launch.manage_config_file` が呼び出されると、`launch` は入力値で config ファイルの内容をパッチします。パッチされた config ファイルはジョブ環境で使用できます。

{{% alert color="secondary" %}}
ジェようとした入力値が確かに使用されるように、ジョブコード内で設定ファイルを読み込む前に `launch.manage_config_file` を呼び出してください。
{{% /alert %}}

### ジョブのローンンチドロワーUIをカスタマイズする

ジョブの入力用のスキーマを定義することで、ジョブをローンンチするためのカスタムUIを作成できます。ジョブのスキーマを定義するには、`launch.manage_wandb_config` または `launch.manage_config_file` への呼び出しにそれを含めます。スキーマは、[JSON Schema](https://json-schema.org/understanding-json-schema/reference) の形をした python 辞書か、Pydantic モデルクラスです。

{{% alert color="secondary" %}}
ジョブ入力スキーマは入力の検証には使用されません。ローンンチドロワーでのUIを定義するためだけに使用されます。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "JSON schema" %}}
次の例では、これらのプロパティを持つスキーマを示します:

- `seed`, 整数
- `trainer`, 一部のキーが指定された辞書:
  - `trainer.learning_rate`, ゼロより大きい必要がある浮動小数点数
  - `trainer.batch_size`, 16、64、または256でなければならない整数
  - `trainer.dataset`, `cifar10` または `cifar100` でなければならない文字列

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
                    "description": "モデルの学習率",
                    "exclusiveMinimum": 0,
                },
                "batch_size": {
                    "type": "integer",
                    "description": "バッチごとのサンプル数",
                    "enum": [16, 64, 256]
                },
                "dataset": {
                    "type": "string",
                    "description": "使用するデータセットの名前",
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

一般的に、以下の JSON スキーマ属性がサポートされています:

| 属性 | 必須 |  注意 |
| --- | --- | --- |
| `type` | はい | `number`, `integer`, `string`, または `object` のいずれかである必要があります |
| `title` | いいえ | プロパティの表示名を上書きします |
| `description` | いいえ | プロパティにヘルパーテキストを提供します |
| `enum` | いいえ | 自由形式のテキスト入力の代わりにドロップダウンの選択を作成します |
| `minimum` | いいえ | `type` が `number` または `integer` である場合のみ許可されます |
| `maximum` | いいえ | `type` が `number` または `integer` である場合のみ許可されます |
| `exclusiveMinimum` | いいえ | `type` が `number` または `integer` である場合のみ許可されます |
| `exclusiveMaximum` | いいえ | `type` が `number` または `integer` である場合のみ許可されます |
| `properties` | いいえ | `type` が `object` の場合、ネストされた設定を定義するために使用されます |
{{% /tab %}}
{{% tab "Pydantic model" %}}
次の例では、これらのプロパティを持つスキーマを示します:

- `seed`, 整数
- `trainer`, 一部のサブ属性が指定されたスキーマ:
  - `trainer.learning_rate`, ゼロより大きい必要がある浮動小数点数
  - `trainer.batch_size`, 1以上256以下でなければならない整数
  - `trainer.dataset`, `cifar10` または `cifar100` でなければならない文字列

```python
class DatasetEnum(str, Enum):
    cifar10 = "cifar10"
    cifar100 = "cifar100"

class Trainer(BaseModel):
    learning_rate: float = Field(gt=0, description="モデルの学習率")
    batch_size: int = Field(ge=1, le=256, description="バッチごとのサンプル数")
    dataset: DatasetEnum = Field(title="Dataset", description="使用するデータセットの名前")

class Schema(BaseModel):
    seed: int
    trainer: Trainer

launch.manage_wandb_config(
    include=["seed", "trainer"],
    exclude=["trainer.private"],
    schema=Schema,
)
```

クラスのインスタンスを使用することもできます:

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

ジョブ入力スキーマを追加すると、ローンンチドロワーに構造化フォームが作成され、ジョブを簡単にローンンチすることができます。

{{< img src="/images/launch/schema_overrides.png" alt="" >}}