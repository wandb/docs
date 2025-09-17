---
title: ジョブの入力を管理する
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-job-inputs
    parent: create-and-deploy-jobs
url: guides/launch/job-inputs
---

Launch のコア体験は、ハイパーパラメーターや Datasets のようなさまざまなジョブ入力を手軽に試し、これらのジョブを適切なハードウェアへルーティングすることです。ジョブが作成されると、元の作者以外のユーザーも W&B の GUI や CLI からこれらの入力を調整できます。CLI または UI から起動する際にジョブ入力をどう設定するかは、[ジョブをキューに追加]({{< relref path="./add-job-to-queue.md" lang="ja" >}}) ガイドを参照してください。

このセクションでは、ジョブで調整可能な入力をプログラム的に制御する方法を説明します。

デフォルトでは、W&B のジョブはジョブの入力として `Run.config` 全体を取得しますが、Launch SDK は run config の特定のキーだけを制御したり、JSON または YAML ファイルを入力として指定したりするための関数を提供します。

{{% alert %}}
Launch SDK の関数を使うには `wandb-core` が必要です。詳しくは [`wandb-core` README](https://github.com/wandb/wandb/blob/main/core/README.md) を参照してください。
{{% /alert %}}

## `Run` オブジェクトを再設定する

ジョブ内で `wandb.init` が返す `Run` オブジェクトは、デフォルトで再設定できます。Launch SDK は、ジョブを起動する際に `Run.config` オブジェクトのどの部分を再設定可能にするかをカスタマイズする方法を提供します。

```python
import wandb
from wandb.sdk import launch

# Launch SDK を使うために必要です。
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
    # など
```

`launch.manage_wandb_config` 関数は、`Run.config` オブジェクトに対する入力値をジョブが受け付けるように設定します。任意の `include` と `exclude` オプションは、入れ子になった config オブジェクト内のパス接頭辞を受け取ります。たとえば、ジョブがエンドユーザーに公開したくないオプションを持つライブラリを使っている場合に便利です。

`include` の接頭辞を指定すると、config 内で `include` の接頭辞に一致するパスのみが入力値を受け付けます。`exclude` の接頭辞を指定すると、`exclude` のリストに一致するパスは入力値から除外されます。あるパスが `include` と `exclude` の両方の接頭辞に一致する場合は、`exclude` の方が優先されます。

上の例では、パス `["trainer.private"]` によって `trainer` オブジェクトから `private` キーが除外され、パス `["trainer"]` によって `trainer` 配下以外のすべてのキーが除外されます。

{{% alert %}}
名前に `.` を含むキーをフィルタリングするには、` \ ` でエスケープした `.` を使用してください。

たとえば、`r"trainer\.private"` は、`trainer` オブジェクト配下の `private` キーではなく、`trainer.private` というキー自体を除外します。

なお、上記の `r` 接頭辞は raw 文字列を表します。
{{% /alert %}}

上記のコードをパッケージ化してジョブとして実行すると、ジョブの入力タイプは次のようになります。

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

W&B の CLI または UI からジョブを起動する際、ユーザーが上書きできるのは `trainer` の 4 つのパラメータだけです。

### Run config 入力にアクセスする

Run config 入力付きで起動したジョブは、`Run.config` を通じて入力値にアクセスできます。ジョブコード内で `wandb.init` が返す `Run` には、入力値が自動的に設定されています。ジョブコードのどこからでも run config の入力値を読み込むには、次を使用してください。
```python
from wandb.sdk import launch

run_config_overrides = launch.load_wandb_config()
```

## ファイルを再設定する

Launch SDK は、ジョブコード内の config ファイルに保存された入力値を管理する方法も提供します。これは多くのディープラーニングや大規模言語モデルのユースケースで一般的なパターンで、たとえばこの [torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_lora.yaml) の例や、この [Axolotl config](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml) などがあります。

{{% alert %}}
[Sweeps on Launch]({{< relref path="../sweeps-on-launch.md" lang="ja" >}}) は、config ファイル入力を sweep パラメータとして使用することをサポートしていません。sweep パラメータは `Run.config` オブジェクトで制御する必要があります。
{{% /alert %}}

`launch.manage_config_file` 関数を使うと、Launch のジョブに config ファイルを入力として追加でき、ジョブを起動する際にそのファイル内の値を編集できます。

デフォルトでは、`launch.manage_config_file` を使用すると run config の入力は取得されません。`launch.manage_wandb_config` を呼び出すと、この挙動を上書きします。

次の例を考えてみましょう。

```python
import yaml
import wandb
from wandb.sdk import launch

# Launch SDK を使うために必要です。
wandb.require("core")

launch.manage_config_file("config.yaml")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

with wandb.init(config=config):
    # など
    pass
```

隣接する `config.yaml` ファイルとともにコードを実行すると仮定します。

```yaml
learning_rate: 0.01
batch_size: 32
model: resnet
dataset: cifar10
```

`launch.manage_config_file` の呼び出しによって、`config.yaml` ファイルがジョブの入力として追加され、W&B の CLI や UI から起動するときに再設定可能になります。

`include` と `exclude` のキーワード引数を使って、`launch.manage_wandb_config` と同様の方法で、その config ファイルに対して許可される入力キーをフィルタリングできます。

### config ファイル入力にアクセスする

Launch によって作成された run で `launch.manage_config_file` が呼び出されると、`launch` は入力値で config ファイルの内容をパッチします。パッチ済みの config ファイルはジョブの環境内で利用できます。

{{% alert color="secondary" %}}
入力値が確実に使用されるよう、ジョブコードで config ファイルを読む前に `launch.manage_config_file` を呼び出してください。
{{% /alert %}}

### ジョブの Launch drawer UI をカスタマイズする

ジョブの入力に対してスキーマを定義すると、ジョブを起動するためのカスタム UI を作成できます。ジョブのスキーマを定義するには、`launch.manage_wandb_config` または `launch.manage_config_file` の呼び出しに含めます。スキーマは、[JSON Schema](https://json-schema.org/understanding-json-schema/reference) 形式の Python 辞書、または Pydantic のモデルクラスのいずれかです。

{{% alert color="secondary" %}}
ジョブ入力スキーマは入力のバリデーションには使用されません。Launch の drawer 内の UI を定義するためだけに使用されます。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "JSON Schema" %}}
次の例は、以下のプロパティを持つスキーマを示します。

- `seed`: 整数
- `trainer`: 次のキーが指定された辞書
  - `trainer.learning_rate`: 0 より大きい必要がある浮動小数
  - `trainer.batch_size`: 16、64、または 256 のいずれかである必要がある整数
  - `trainer.dataset`: `cifar10` または `cifar100` のいずれかである必要がある文字列

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

| Attribute | Required | Notes |
| --- | --- | --- |
| `type` | Yes | `number`、`integer`、`string`、`object` のいずれかである必要があります |
| `title` | No | プロパティの表示名を上書きします |
| `description` | No | プロパティの補足テキストを表示します |
| `enum` | No | 自由入力ではなくプルダウン選択を作成します |
| `minimum` | No | `type` が `number` または `integer` の場合のみ使用可能 |
| `maximum` | No | `type` が `number` または `integer` の場合のみ使用可能 |
| `exclusiveMinimum` | No | `type` が `number` または `integer` の場合のみ使用可能 |
| `exclusiveMaximum` | No | `type` が `number` または `integer` の場合のみ使用可能 |
| `properties` | No | `type` が `object` の場合、入れ子の設定を定義するために使用します |
{{% /tab %}}
{{% tab "Pydantic モデル" %}}
次の例は、以下のプロパティを持つスキーマを示します。

- `seed`: 整数
- `trainer`: 次のサブ属性が指定されたスキーマ
  - `trainer.learning_rate`: 0 より大きい必要がある浮動小数
  - `trainer.batch_size`: 1 以上 256 以下である必要がある整数
  - `trainer.dataset`: `cifar10` または `cifar100` のいずれかである必要がある文字列

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

クラスのインスタンスを使うこともできます。

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

ジョブ入力スキーマを追加すると、Launch の drawer に構造化フォームが作成され、ジョブを簡単に起動できるようになります。

{{< img src="/images/launch/schema_overrides.png" alt="ジョブ入力スキーマのフォーム" >}}