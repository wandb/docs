---
title: ジョブ入力を管理する
menu:
  launch:
    identifier: job-inputs
    parent: create-and-deploy-jobs
url: guides/launch/job-inputs
---

Launch のコア体験は、ハイパーパラメーターやデータセットなどの異なるジョブ入力を簡単に実験し、それらのジョブを適切なハードウェアへルーティングすることです。一度ジョブが作成されると、元の作成者以外のユーザーも W&B の GUI や CLI を通じてこれらの入力を調整できます。CLI や UI からのジョブ入力の設定方法については、[Enqueue jobs]({{< relref "./add-job-to-queue.md" >}}) ガイドを参照してください。

このセクションでは、プログラムでジョブの調整可能な入力を制御する方法について説明します。

デフォルトでは、W&B のジョブは `Run.config` 全体をジョブへの入力としてキャプチャしますが、Launch SDK は run config の特定のキーを選択制御したり、JSON や YAML ファイルを入力として指定する関数を提供します。

{{% alert %}}
Launch SDK の関数は `wandb-core` が必要です。詳細は [`wandb-core` README](https://github.com/wandb/wandb/blob/main/core/README.md) をご覧ください。
{{% /alert %}}

## `Run` オブジェクトの再設定

ジョブ内で `wandb.init` から返される `Run` オブジェクトは、デフォルトで再設定が可能です。Launch SDK を使うと、ジョブの起動時にどの `Run.config` オブジェクトの部分を再設定できるかをカスタマイズできます。

```python
import wandb
from wandb.sdk import launch

# launch sdk の利用に必須
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
    # その他
```

`launch.manage_wandb_config` 関数は、`Run.config` オブジェクトの入力値をジョブで受け付けるように設定します。オプションの `include`, `exclude` はネストした config オブジェクト内でのパスのプレフィックスをとります。たとえば、特定のライブラリのオプションをエンドユーザーに公開したくない場合に役立ちます。

`include` プレフィックスが指定されていれば、そのプレフィックスに一致する config 内のパスのみが入力値として受け付けられます。`exclude` プレフィックスが指定されると、`exclude` リストに一致するパスは入力値から除外されます。もし同時に `include` と `exclude` のプレフィックス両方に一致した場合は、`exclude` が優先されます。

上記の例では、パス `["trainer.private"]` で `trainer` オブジェクト内の `private` キーが、パス `["trainer"]` で `trainer` オブジェクト以外の全てのキーが除外されます。

{{% alert %}}
キー名に `.` が含まれている場合は、`\` でエスケープしてください。

例：`r"trainer\.private"` は、`trainer` オブジェクト内の `private` キーではなく、`trainer.private` というキーを除外します。

なお、上記の `r` プレフィックスは生文字列（raw string）を意味します。
{{% /alert %}}

上記のコードをパッケージ化してジョブとして実行した場合、ジョブの入力型は下記のようになります。

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

W&B CLI や UI からジョブを起動する際、ユーザーが上記4つの `trainer` パラメータだけを上書きできるようになります。

### run config 入力値へのアクセス

run config 入力でジョブを起動した場合、`Run.config` 経由で入力値にアクセスできます。ジョブで `wandb.init` から返される `Run` は、入力値が自動でセットされています。
また、ジョブコード内でどこでも
```python
from wandb.sdk import launch

run_config_overrides = launch.load_wandb_config()
```
と書くことで、run config の入力値をロードできます。

## ファイルを再設定

Launch SDK はまた、ジョブコード内の設定ファイルに保存された入力値を管理する方法も提供します。これは多くのディープラーニングや大規模言語モデルのユースケース (例: [torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_lora.yaml) や [Axolotl config](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml)) でよく利用されるパターンです。

{{% alert %}}
[Sweeps on Launch]({{< relref "../sweeps-on-launch.md" >}}) では、設定ファイルをスイープパラメータとして利用できません。sweep パラメータは `Run.config` オブジェクト経由で制御される必要があります。
{{% /alert %}}

`launch.manage_config_file` 関数を使えば、設定ファイル全体を Launch ジョブの入力として追加でき、ジョブの起動時に設定ファイル内の値を編集可能にできます。

`launch.manage_config_file` を使った場合、デフォルトでは run config の入力はキャプチャされません。`launch.manage_wandb_config` を呼び出すことでこの挙動を上書きできます。

以下の例を参考にしてください。

```python
import yaml
import wandb
from wandb.sdk import launch

# launch sdk の利用に必須
wandb.require("core")

launch.manage_config_file("config.yaml")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

with wandb.init(config=config):
    # その他
    pass
```

このコードと隣接する `config.yaml` ファイルが存在するとします:

```yaml
learning_rate: 0.01
batch_size: 32
model: resnet
dataset: cifar10
```

`launch.manage_config_file` を呼び出すことで、`config.yaml` ファイル全体がジョブの入力として追加され、W&B CLI や UI から起動時に値の変更が可能になります。

`include` および `exclude` キーワード引数は、`launch.manage_wandb_config` と同様に、設定ファイルに対して許容される入力キーをフィルタリングできます。

### 設定ファイル入力へのアクセス

Launch で作成した run 内で `launch.manage_config_file` が呼ばれると、`launch` が設定ファイルの内容を入力値でパッチします。パッチ済みの設定ファイルは、ジョブの環境内で使用できます。

{{% alert color="secondary" %}}
ジョブコード内で設定ファイルを読む前に `launch.manage_config_file` を呼び出してください。これにより入力値が反映されます。
{{% /alert %}}

### ジョブの launch ドロワー UI カスタマイズ

ジョブの入力のスキーマを定義することで、ジョブの起動 UI をカスタマイズできます。ジョブのスキーマは、`launch.manage_wandb_config` または `launch.manage_config_file` の呼び出し時に指定します。スキーマは [JSON Schema](https://json-schema.org/understanding-json-schema/reference) 形式の python dict か、Pydantic モデルクラスで指定可能です。

{{% alert color="secondary" %}}
ジョブ入力スキーマは入力値のバリデーションには使用されません。launch ドロワーの UI 定義のみに使用されます。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "JSON schema" %}}
以下の例は、次のプロパティを持つスキーマを示します：

- `seed`: 整数
- `trainer`: 一部のキーを指定した辞書
  - `trainer.learning_rate`: 0より大きい float
  - `trainer.batch_size`: 16, 64, 256 のいずれかの整数
  - `trainer.dataset`: `cifar10` または `cifar100` のいずれかの文字列

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
                    "description": "利用するデータセット名",
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

| 属性 | 必須 | 備考 |
| --- | --- | --- |
| `type` | はい | `number`, `integer`, `string`, `object` のいずれか |
| `title` | いいえ | プロパティ名の表示を上書き |
| `description` | いいえ | プロパティの補助説明を追加 |
| `enum` | いいえ | 自由入力ではなく選択式のドロップダウンを生成 |
| `minimum` | いいえ | `type` が `number` または `integer` の場合のみ許可 |
| `maximum` | いいえ | `type` が `number` または `integer` の場合のみ許可 |
| `exclusiveMinimum` | いいえ | `type` が `number` または `integer` の場合のみ許可 |
| `exclusiveMaximum` | いいえ | `type` が `number` または `integer` の場合のみ許可 |
| `properties` | いいえ | `type` が `object` の場合、ネストした設定を定義 |
{{% /tab %}}
{{% tab "Pydantic model" %}}
以下の例は、次のプロパティを持つスキーマを示します：

- `seed`: 整数
- `trainer`: 一部のサブ属性を指定したスキーマ
  - `trainer.learning_rate`: 0より大きい float
  - `trainer.batch_size`: 1以上256以下の整数
  - `trainer.dataset`: `cifar10` または `cifar100` のいずれかの文字列

```python
class DatasetEnum(str, Enum):
    cifar10 = "cifar10"
    cifar100 = "cifar100"

class Trainer(BaseModel):
    learning_rate: float = Field(gt=0, description="モデルの学習率")
    batch_size: int = Field(ge=1, le=256, description="バッチごとのサンプル数")
    dataset: DatasetEnum = Field(title="データセット", description="利用するデータセット名")

class Schema(BaseModel):
    seed: int
    trainer: Trainer

launch.manage_wandb_config(
    include=["seed", "trainer"],
    exclude=["trainer.private"],
    schema=Schema,
)
```

クラスのインスタンスも利用可能です：

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

ジョブ入力スキーマを追加すると、launch ドロワー内に構造化フォームが生成され、ジョブの起動がより簡単になります。

{{< img src="/images/launch/schema_overrides.png" alt="Job input schema form" >}}