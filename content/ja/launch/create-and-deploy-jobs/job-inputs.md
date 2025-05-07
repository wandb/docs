---
title: ジョブ入力を管理する
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-job-inputs
    parent: create-and-deploy-jobs
url: /ja/guides/launch/job-inputs
---

Launch のコア体験は、ハイパーパラメーターやデータセットのような異なるジョブの入力を簡単に実験し、それらのジョブを適切なハードウェアにルーティングすることです。一度ジョブが作成されると、元の作成者以外のユーザーも W&B GUI または CLI を介してこれらの入力を調整できます。CLI または UI からジョブ入力を設定する方法については、[Enqueue jobs]({{< relref path="./add-job-to-queue.md" lang="ja" >}}) ガイドを参照してください。

このセクションでは、プログラム的にジョブの調整可能な入力を制御する方法を説明します。

デフォルトでは、W&B ジョブは `Run.config` 全体をジョブの入力としてキャプチャしますが、Launch SDK は run config の選択したキーを制御したり、JSON または YAML ファイルを入力として指定するための機能を提供します。

{{% alert %}}
Launch SDK の関数は `wandb-core` を必要とします。詳細については、[`wandb-core` README](https://github.com/wandb/wandb/blob/main/core/README.md) を参照してください。
{{% /alert %}}

## `Run` オブジェクトの再設定

ジョブ内の `wandb.init` によって返される `Run` オブジェクトは、デフォルトで再設定可能です。Launch SDK は、ジョブの Launch 時に `Run.config` オブジェクトのどの部分が再設定可能かをカスタマイズする方法を提供します。

```python
import wandb
from wandb.sdk import launch

# Launch SDK の使用に必要
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
    # 等
```

関数 `launch.manage_wandb_config` は、`Run.config` オブジェクトに対する入力値をジョブに受け入れるように設定します。オプションの `include` および `exclude` オプションは、ネストされた config オブジェクト内のパスプレフィクスを取ります。例えば、ジョブがエンドユーザーに公開したくないオプションを持つライブラリを使用している場合に役立ちます。

`include` プレフィクスが提供されている場合、config 内で `include` プレフィクスと一致するパスのみが入力値を受け入れます。`exclude` プレフィクスが提供されている場合、`exclude` リストと一致するパスは入力値からフィルタリングされません。あるパスが `include` および `exclude` プレフィクスの両方に一致する場合、`exclude` プレフィクスが優先されます。

前述の例では、パス `["trainer.private"]` は `trainer` オブジェクトから `private` キーをフィルタリングし、パス `["trainer"]` は `trainer` オブジェクト下のすべてのキー以外をフィルタリングします。

{{% alert %}}
名前に `.` を含むキーをフィルタリングするには、`\`-エスケープされた `.` を使用します。

例えば、`r"trainer\.private"` は `trainer` オブジェクト下の `private` キーではなく `trainer.private` キーをフィルタリングします。

上記の `r` プレフィクスは生文字列を示すことに注意してください。
{{% /alert %}}

上記のコードがパッケージ化され、ジョブとして実行される場合、ジョブの入力タイプは次のようになります：

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

W&B CLI または UI からジョブをローンチする際、ユーザーは 4 つの `trainer` パラメーターのみを上書きできます。

### run config の入力へのアクセス

run config の入力でローンチされたジョブは、`Run.config` を通じて入力値にアクセスできます。ジョブ コード内で `wandb.init` によって返された `Run` は、入力値を自動的に設定します。ジョブ コードのどこでも run config の入力値をロードするには、次を使用します：

```python
from wandb.sdk import launch

run_config_overrides = launch.load_wandb_config()
```

## ファイルの再設定

Launch SDK はまた、ジョブ コードで設定ファイルに格納された入力値を管理する方法も提供します。これは、多くのディープラーニングや大規模な言語モデルのユースケースで一般的なパターンであり、例えばこの [torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_lora.yaml) の例やこの [Axolotl config](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml) などがあります。

{{% alert %}}
[Sweeps on Launch]({{< relref path="../sweeps-on-launch.md" lang="ja" >}}) は、sweep パラメーターとしての設定ファイル入力の使用をサポートしていません。sweep パラメーターは `Run.config` オブジェクトで制御する必要があります。
{{% /alert %}}

`launch.manage_config_file` 関数を使用して、設定ファイルを Launch ジョブの入力として追加し、ジョブをローンチする際に設定ファイル内の値の編集アクセスを提供します。

デフォルトでは、`launch.manage_config_file` が使用されると run config 入力はキャプチャされません。`launch.manage_wandb_config` を呼び出すことで、この振る舞いを上書きします。

以下の例を考えてみましょう：

```python
import yaml
import wandb
from wandb.sdk import launch

# Launch SDK の使用に必要
wandb.require("core")

launch.manage_config_file("config.yaml")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

with wandb.init(config=config):
    # 等
    pass
```

コードが隣接するファイル `config.yaml` と一緒に実行されていると想像してください：

```yaml
learning_rate: 0.01
batch_size: 32
model: resnet
dataset: cifar10
```

`launch.manage_config_file` の呼び出しは、`config.yaml` ファイルをジョブの入力として追加し、W&B CLI または UI からローンチする際に再設定可能にします。

`launch.manage_wandb_config` と同じようにして、設定ファイルに対する許容入力キーをフィルタリングするために `include` と `exclude` のキーワード引数を使用できます。

### 設定ファイルの入力へのアクセス

Launch によって作成された run で `launch.manage_config_file` が呼び出されると、`launch` は設定ファイルの内容を入力値でパッチします。修正された設定ファイルはジョブ 環境で利用可能です。

{{% alert color="secondary" %}}
ジョブ コードで最初に設定ファイルを読み取る前に `launch.manage_config_file` を呼び出すことで、入力値が使用されることを確実にします。
{{% /alert %}}


### ジョブの launch ドロワー UI のカスタマイズ

ジョブの入力スキーマを定義することで、ジョブをローンチするためのカスタム UI を作成できます。ジョブのスキーマを定義するには、`launch.manage_wandb_config` または `launch.manage_config_file` の呼び出しにそれを含めます。スキーマは、[JSON Schema](https://json-schema.org/understanding-json-schema/reference) の形での Python 辞書または Pydantic モデル クラスのいずれかです。

{{% alert color="secondary" %}}
ジョブ入力スキーマは入力を検証するために使用されません。それらは launch ドロワー内の UI を定義するためだけに使用されます。
{{% /alert %}}


{{< tabpane text=true >}}
{{% tab "JSON schema" %}}
次の例は、以下のプロパティを持つスキーマを示しています：

- `seed`, 整数
- `trainer`, いくつかのキーが指定された辞書：
  - `trainer.learning_rate`, ゼロより大きくなければならない浮動小数点数
  - `trainer.batch_size`, 16, 64, 256 のいずれかでなければならない整数
  - `trainer.dataset`, `cifar10` または `cifar100` のいずれかでなければならない文字列

```python
schema = {
    "type": "object",
    "properties": {
        "seed": {
          "type": "integer"
        },
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

一般的に、以下の JSON Schema 属性がサポートされています：

| 属性 | 必須 |  注釈 |
| --- | --- | --- |
| `type` | Yes | `number`, `integer`, `string`, `object` のいずれかでなければなりません |
| `title` | No | プロパティの表示名を上書きします |
| `description` | No | プロパティのヘルプテキストを提供します |
| `enum` | No | フリーフォームのテキスト入力の代わりにドロップダウン選択を作成します |
| `minimum` | No | `type` が `number` または `integer` のときのみ許可されます |
| `maximum` | No | `type` が `number` または `integer` のときのみ許可されます |
| `exclusiveMinimum` | No | `type` が `number` または `integer` のときのみ許可されます |
| `exclusiveMaximum` | No | `type` が `number` または `integer` のときのみ許可されます |
| `properties` | No | `type` が `object` のとき、ネストされた設定を定義するために使用します |
{{% /tab %}}
{{% tab "Pydantic model" %}}
次の例は、以下のプロパティを持つスキーマを示しています：

- `seed`, 整数
- `trainer`, いくつかのサブ属性が指定されたスキーマ：
  - `trainer.learning_rate`, ゼロより大きくなければならない浮動小数点数
  - `trainer.batch_size`, 1 から 256 までの範囲でなければならない整数
  - `trainer.dataset`, `cifar10` または `cifar100` のいずれかでなければならない文字列

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

クラスのインスタンスを使用することもできます：

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

ジョブ入力スキーマを追加すると、launch ドロワーに構造化されたフォームが作成され、ジョブのローンチが容易になります。

{{< img src="/images/launch/schema_overrides.png" alt="" >}}