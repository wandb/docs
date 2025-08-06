---
title: ジョブ入力の管理
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-job-inputs
    parent: create-and-deploy-jobs
url: guides/launch/job-inputs
---

Launch のコア体験は、ハイパーパラメーターやデータセットなど、さまざまなジョブ入力を手軽に変更し、それらのジョブを適切なハードウェアに割り当てることです。一度ジョブが作成されると、元の作成者以外のユーザーも W&B GUI や CLI を通じてこれらの入力を調整できます。CLI や UI からジョブの入力値を設定する方法については、[ジョブをキューに追加]({{< relref path="./add-job-to-queue.md" lang="ja" >}})ガイドをご覧ください。

このセクションでは、ジョブの入力値をプログラム的に操作する方法について説明します。

デフォルトでは、W&B ジョブは `Run.config` 全体をジョブの入力としてキャプチャしますが、Launch SDK を使うと run config の特定のキーだけを選択したり、JSON や YAML ファイルを入力として指定したりできる関数が提供されています。

{{% alert %}}
Launch SDK の関数を使うには `wandb-core` が必要です。詳細は [`wandb-core` の README](https://github.com/wandb/wandb/blob/main/core/README.md) をご覧ください。
{{% /alert %}}

## `Run` オブジェクトの再設定

ジョブ内で `wandb.init` が返す `Run` オブジェクトは、デフォルトで再設定可能です。Launch SDK を使うと、ジョブ起動時にどの `Run.config` オブジェクトの部分を再設定できるかをカスタマイズできます。

```python
import wandb
from wandb.sdk import launch

# Launch SDK の利用に必須
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

`launch.manage_wandb_config` 関数は、ジョブが `Run.config` オブジェクトに対して入力値を受け取れるよう設定します。オプションの `include` と `exclude` には、ネストされた設定オブジェクト内のパスのプレフィックスを渡せます。たとえば、特定のライブラリのオプションをエンドユーザーに公開したくない場合に便利です。

`include` プレフィックスが指定された場合、そのプレフィックスに一致するパスのみが入力値を受け付けます。`exclude` プレフィックスが指定された場合は、該当するパスが入力値として除外されます。同じパスが `include` と `exclude` の両方に一致する場合は、`exclude` が優先されます。

前述の例では、`["trainer.private"]` というパスで `trainer` オブジェクトの `private` キーを除外し、`["trainer"]` のパスで `trainer` オブジェクト配下以外のキーを除外します。

{{% alert %}}
名前に `.` を含むキーを除外する場合は、`\` を使ってエスケープしてください。

例えば、`r"trainer\.private"` とすると、`trainer` オブジェクト内の `private` ではなく、`trainer.private` という名前のキーを除外します。

なお、上記で使っている `r` プレフィックスは Python の raw 文字列を表しています。
{{% /alert %}}

このコードがパッケージ化されてジョブとして実行されると、ジョブの入力タイプは次のようになります：

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

W&B の CLI または UI からジョブを起動すると、ユーザーは `trainer` の4つのパラメータのみを上書きできます。

### run config 入力値の取得

run config の入力値付きで起動されたジョブは、`Run.config` を通じて値にアクセスできます。ジョブコード内で `wandb.init` が返す `Run` には、入力値が自動でセットされています。ジョブ内の任意の場所で
```python
from wandb.sdk import launch

run_config_overrides = launch.load_wandb_config()
```
のようにして run config の入力値を取得できます。

## ファイルの再設定

Launch SDK では、ジョブコード内の設定ファイルとして保存された入力値を管理する方法も提供しています。これは多くのディープラーニングや大規模言語モデルのユースケースでよく使われるパターンです（[torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_lora.yaml) の例や [Axolotl config](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml) など）。

{{% alert %}}
[Launch での Sweeps]({{< relref path="../sweeps-on-launch.md" lang="ja" >}}) では、設定ファイルの入力値をスイープパラメーターとして利用することはできません。スイープパラメーターは必ず `Run.config` オブジェクト経由で制御してください。
{{% /alert %}}

`launch.manage_config_file` 関数を使うと、設定ファイルを Launch ジョブの入力として追加でき、ジョブ起動時にその設定ファイル内の値を編集できるようになります。

デフォルトでは、`launch.manage_config_file` を使うと run config 入力値はキャプチャされません。`launch.manage_wandb_config` を呼び出すとこの振る舞いを上書きします。

例を見てみましょう：

```python
import yaml
import wandb
from wandb.sdk import launch

# Launch SDK の利用に必須
wandb.require("core")

launch.manage_config_file("config.yaml")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

with wandb.init(config=config):
    # など
    pass
```

このコードが、隣接する `config.yaml` ファイルとともに実行されたと仮定します。

```yaml
learning_rate: 0.01
batch_size: 32
model: resnet
dataset: cifar10
```

`launch.manage_config_file` を呼ぶことで、`config.yaml` ファイルがジョブの入力として追加され、W&B CLI や UI から起動する際に再設定できるようになります。

`include` および `exclude` キーワード引数は、`launch.manage_wandb_config` と同様に、設定ファイル内の入力キーを制限するために使えます。

### 設定ファイル入力値の取得

Launch で作成した run で `launch.manage_config_file` を呼ぶと、`launch` が設定ファイルの内容を入力値でパッチします。このパッチ済みファイルはジョブの環境内で利用できます。

{{% alert color="secondary" %}}
ジョブコード内で設定ファイルを読む前に `launch.manage_config_file` を必ず呼び、入力値が反映されるようにしてください。
{{% /alert %}}

### ジョブの launch drawer UI をカスタマイズ

ジョブの入力用スキーマを定義することで、ジョブ起動時のカスタム UI を作成できます。スキーマは `launch.manage_wandb_config` または `launch.manage_config_file` を呼ぶ際に指定します。スキーマは [JSON Schema](https://json-schema.org/understanding-json-schema/reference) 形式の Python dict または Pydantic モデルクラスで指定できます。

{{% alert color="secondary" %}}
ジョブ入力のスキーマは値のバリデーションには使われません。launch drawer 上の UI を定義するためだけに使用されます。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "JSON schema" %}}
次の例では、以下のプロパティを定義したスキーマを示しています：

- `seed`：整数
- `trainer`：以下のキーを持つ辞書
  - `trainer.learning_rate`：0より大きい float
  - `trainer.batch_size`：16, 64, 256 のいずれかの整数
  - `trainer.dataset`：`cifar10` または `cifar100` のいずれかの文字列

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
                    "description": "バッチあたりのサンプル数",
                    "enum": [16, 64, 256]
                },
                "dataset": {
                    "type": "string",
                    "description": "使用するデータセット名",
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

一般的に、サポートされている JSON Schema 属性は次の通りです：

| 属性名 | 必須 | メモ |
| --- | --- | --- |
| `type` | Yes | `number`、`integer`、`string`、`object` のいずれか |
| `title` | No | プロパティの表示名を上書き |
| `description` | No | プロパティの補足テキスト |
| `enum` | No | 自由記述のテキスト入力の代わりにプルダウン選択を作成 |
| `minimum` | No | `type` が `number` または `integer` の場合にのみ許可 |
| `maximum` | No | `type` が `number` または `integer` の場合にのみ許可 |
| `exclusiveMinimum` | No | `type` が `number` または `integer` の場合にのみ許可 |
| `exclusiveMaximum` | No | `type` が `number` または `integer` の場合にのみ許可 |
| `properties` | No | `type` が `object` の場合、ネストされた設定を定義 |
{{% /tab %}}
{{% tab "Pydantic model" %}}
次の例では、以下のプロパティを定義したスキーマを示しています：

- `seed`：整数
- `trainer`：以下のサブ属性を持つスキーマ
  - `trainer.learning_rate`：0より大きい float
  - `trainer.batch_size`：1～256 の範囲の整数
  - `trainer.dataset`：`cifar10` または `cifar100` のいずれかの文字列

```python
class DatasetEnum(str, Enum):
    cifar10 = "cifar10"
    cifar100 = "cifar100"

class Trainer(BaseModel):
    learning_rate: float = Field(gt=0, description="モデルの学習率")
    batch_size: int = Field(ge=1, le=256, description="バッチあたりのサンプル数")
    dataset: DatasetEnum = Field(title="データセット", description="使用するデータセット名")

class Schema(BaseModel):
    seed: int
    trainer: Trainer

launch.manage_wandb_config(
    include=["seed", "trainer"],
    exclude=["trainer.private"],
    schema=Schema,
)
```

クラスインスタンスも指定可能です：

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

ジョブ入力用スキーマを追加すると、launch drawer に構造化されたフォームが作成され、より手軽にジョブを起動できます。

{{< img src="/images/launch/schema_overrides.png" alt="Job input schema form" >}}