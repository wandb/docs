---
title: Run
menu:
  reference:
    identifier: ja-ref-python-run
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L461-L4042 >}}

wandb によってログされる計算の単位。通常、これは 機械学習 の実験です。

```python
Run(
    settings: Settings,
    config: (dict[str, Any] | None) = None,
    sweep_config: (dict[str, Any] | None) = None,
    launch_config: (dict[str, Any] | None) = None
) -> None
```

`wandb.init()` で run を作成します。

```python
import wandb

run = wandb.init()
```

プロセス内にアクティブな `wandb.Run` は最大で 1 つだけであり、
`wandb.run` としてアクセスできます。

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```

`wandb.log` でログに記録した内容はすべて、その run に送信されます。

同じスクリプトまたは notebook で複数の run を開始する場合は、
実行中の run を終了する必要があります。Run は `wandb.finish` で終了するか、
`with` ブロックで使用することで終了できます。

```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
    pass  # log data here

assert wandb.run is None
```

run の作成の詳細については、`wandb.init` のドキュメントを参照するか、
[`wandb.init` のガイド](https://docs.wandb.ai/guides/track/launch) を確認してください。

分散トレーニングでは、ランク 0 のプロセスで単一の run を作成し、
そのプロセスからのみ情報をログに記録するか、各プロセスで run を作成し、
それぞれから個別にログに記録し、`wandb.init` への `group` 引数を使用して結果をグループ化できます。
W&B を使用した分散トレーニングの詳細については、
[ガイド](https://docs.wandb.ai/guides/track/log/distributed-training) を確認してください。

現在、`wandb.Api` には並列 `Run` オブジェクトがあります。最終的には、これら 2 つの
オブジェクトはマージされます。

| 属性 |  |
| :--- | :--- |
|  `summary` |  (Summary) 各 `wandb.log()` キーに設定された単一の値。デフォルトでは、summary は最後にログに記録された値に設定されます。最終値の代わりに、最大精度のような最良の値に summary を手動で設定できます。 |
|  `config` |  この run に関連付けられた Config オブジェクト。 |
|  `dir` |  run に関連付けられたファイルが保存されるディレクトリー。 |
|  `entity` |  run に関連付けられた W&B のエンティティーの名前。エンティティーは、ユーザー名、チーム名、または組織名にすることができます。 |
|  `group` |  run に関連付けられたグループの名前。グループを設定すると、W&B UI で run をわかりやすい方法で整理できます。分散トレーニングを行っている場合は、トレーニング内のすべての run に同じグループを指定する必要があります。クロスバリデーションを行っている場合は、すべてのクロスバリデーションの fold に同じグループを指定する必要があります。 |
|  `id` |  この run の識別子。 |
|  `mode` |  `0.9.x` 以前との互換性のため。最終的には非推奨になります。 |
|  `name` |  run の表示名。表示名は一意であるとは限らず、記述的な場合があります。デフォルトでは、ランダムに生成されます。 |
|  `notes` |  run に関連付けられたメモ (存在する場合)。メモは複数行の文字列にすることができ、`$$` 内で markdown および latex 数式を使用することもできます (例: `$x + 3$`)。 |
|  `path` |  run へのパス。run パスには、エンティティー、プロジェクト、run ID が `entity/project/run_id` の形式で含まれます。 |
|  `project` |  run に関連付けられた W&B プロジェクトの名前。 |
|  `resumed` |  run が再開された場合は True、そうでない場合は False。 |
|  `settings` |  run の Settings オブジェクトのフリーズされたコピー。 |
|  `start_time` |  run が開始されたときの Unix タイムスタンプ (秒単位)。 |
|  `starting_step` |  run の最初のステップ。 |
|  `step` |  ステップの現在の値。このカウンターは `wandb.log` によってインクリメントされます。 |
|  `sweep_id` |  run に関連付けられた sweep の ID (存在する場合)。 |
|  `tags` |  run に関連付けられたタグ (存在する場合)。 |
|  `url` |  run に関連付けられた W&B の URL。 |

## メソッド

### `alert`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3567-L3600)

```python
alert(
    title: str,
    text: str,
    level: (str | AlertLevel | None) = None,
    wait_duration: (int | float | timedelta | None) = None
) -> None
```

指定されたタイトルとテキストでアラートを起動します。

| Args |  |
| :--- | :--- |
|  `title` |  (str) アラートのタイトル。64 文字未満である必要があります。 |
|  `text` |  (str) アラートのテキスト本文。 |
|  `level` |  (str または AlertLevel、オプション) 使用するアラートレベル。`INFO`、`WARN`、または `ERROR` のいずれかです。 |
|  `wait_duration` |  (int、float、または timedelta、オプション) このタイトルで別のアラートを送信するまでの待機時間 (秒単位)。 |

### `define_metric`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2660-L2721)

```python
define_metric(
    name: str,
    step_metric: (str | wandb_metric.Metric | None) = None,
    step_sync: (bool | None) = None,
    hidden: (bool | None) = None,
    summary: (str | None) = None,
    goal: (str | None) = None,
    overwrite: (bool | None) = None
) -> wandb_metric.Metric
```

`wandb.log()` でログに記録されたメトリクスをカスタマイズします。

| Args |  |
| :--- | :--- |
|  `name` |  カスタマイズするメトリクスの名前。 |
|  `step_metric` |  自動生成されたグラフで、このメトリクスの X 軸として機能する別のメトリクスの名前。 |
|  `step_sync` |  step_metric が明示的に指定されていない場合、step_metric の最後の値を自動的に `run.log()` に挿入します。step_metric が指定されている場合、デフォルトは True です。 |
|  `hidden` |  このメトリクスを自動プロットから非表示にします。 |
|  `summary` |  summary に追加される集計メトリクスを指定します。サポートされている集計には、"min"、"max"、"mean"、"last"、"best"、"copy"、および "none" があります。"best" は goal パラメータとともに使用されます。"none" は summary が生成されないようにします。"copy" は非推奨であり、使用しないでください。 |
|  `goal` |  "best" summary タイプを解釈する方法を指定します。サポートされているオプションは、"minimize" と "maximize" です。 |
|  `overwrite` |  false の場合、この呼び出しは、指定されていないパラメータにそれらの値を使用して、同じメトリクスの以前の `define_metric` 呼び出しとマージされます。true の場合、指定されていないパラメータは、以前の呼び出しで指定された値を上書きします。 |

| 戻り値 |  |
| :--- | :--- |
|  この呼び出しを表すオブジェクト。それ以外の場合は破棄できます。 |

### `detach`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2885-L2886)

```python
detach() -> None
```

### `display`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1219-L1236)

```python
display(
    height: int = 420,
    hidden: bool = (False)
) -> bool
```

この run を jupyter で表示します。

### `finish`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2075-L2106)

```python
finish(
    exit_code: (int | None) = None,
    quiet: (bool | None) = None
) -> None
```

run を終了し、残りのデータをアップロードします。

W&B run の完了をマークし、すべてのデータがサーバーに同期されるようにします。
run の最終状態は、終了条件と同期ステータスによって決まります。

#### Run の状態:

- Running: データをログに記録している、またはハートビートを送信しているアクティブな run。
- Crashed: ハートビートの送信が予期せず停止した run。
- Finished: すべてのデータが同期されて正常に完了した run (`exit_code=0`)。
- Failed: エラーで完了した run (`exit_code!=0`)。

| Args |  |
| :--- | :--- |
|  `exit_code` |  run の終了ステータスを示す整数。成功の場合は 0 を使用し、他の値を使用すると run が失敗としてマークされます。 |
|  `quiet` |  非推奨。`wandb.Settings(quiet=...)` を使用して、ログの冗長性を構成します。 |

### `finish_artifact`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3163-L3215)

```python
finish_artifact(
    artifact_or_path: (Artifact | str),
    name: (str | None) = None,
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    distributed_id: (str | None) = None
) -> Artifact
```

run の出力として、非 final な artifact を終了します。

同じ分散 ID を使用した後続の "upsert" は、新しいバージョンになります。

| Args |  |
| :--- | :--- |
|  `artifact_or_path` |  (str または Artifact) この artifact のコンテンツへのパス。次の形式にすることができます。- `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` `wandb.Artifact` を呼び出して作成された Artifact オブジェクトを渡すこともできます。 |
|  `name` |  (str、オプション) artifact 名。エンティティー/プロジェクトでプレフィックスを付けることができます。有効な名前は、次の形式にすることができます。- name:version - name:alias - digest これが指定されていない場合、パスの basename に現在の run ID が付加されたものがデフォルトになります。 |
|  `type` |  (str) ログに記録する artifact のタイプ (例: `dataset`、`model`) |
|  `aliases` |  (list、オプション) この artifact に適用するエイリアス。デフォルトは `["latest"]` です。 |
|  `distributed_id` |  (string、オプション) すべての分散ジョブが共有する一意の文字列。None の場合、run のグループ名がデフォルトになります。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `get_project_url`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1091-L1099)

```python
get_project_url() -> (str | None)
```

run に関連付けられた W&B プロジェクトの URL を返します (存在する場合)。

オフライン run にはプロジェクト URL がありません。

### `get_sweep_url`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1101-L1106)

```python
get_sweep_url() -> (str | None)
```

run に関連付けられた sweep の URL を返します (存在する場合)。

### `get_url`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1108-L1116)

```python
get_url() -> (str | None)
```

W&B run の URL を返します (存在する場合)。

オフライン run には URL がありません。

### `join`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2148-L2159)

```python
join(
    exit_code: (int | None) = None
) -> None
```

`finish()` の非推奨のエイリアス - 代わりに finish を使用してください。

### `link_artifact`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2888-L2951)

```python
link_artifact(
    artifact: Artifact,
    target_path: str,
    aliases: (list[str] | None) = None
) -> None
```

指定された artifact をポートフォリオ (artifact の昇格されたコレクション) にリンクします。

リンクされた artifact は、指定されたポートフォリオの UI に表示されます。

| Args |  |
| :--- | :--- |
|  `artifact` |  リンクされる (パブリックまたはローカル) artifact |
|  `target_path` |  `str` - 次の形式を取ります: `{portfolio}`、`{project}/{portfolio}`、または `{entity}/{project}/{portfolio}` |
|  `aliases` |  `List[str]` - オプションのエイリアス。このリンクされた artifact のポートフォリオ内でのみ適用されます。エイリアス "latest" は、リンクされている artifact の最新バージョンに常に適用されます。 |

| 戻り値 |  |
| :--- | :--- |
|  なし |

### `link_model`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3466-L3565)

```python
link_model(
    path: StrPath,
    registered_model_name: str,
    name: (str | None) = None,
    aliases: (list[str] | None) = None
) -> None
```

モデル artifact バージョンをログに記録し、モデルレジストリ内の登録済みモデルにリンクします。

リンクされたモデルバージョンは、指定された登録済みモデルの UI に表示されます。

#### 手順:

- 'name' モデル artifact がログに記録されているかどうかを確認します。ログに記録されている場合は、'path' にあるファイルと一致する artifact バージョンを使用するか、新しいバージョンをログに記録します。それ以外の場合は、'path' の下のファイルを新しいモデル artifact 'name'、タイプ 'model' としてログに記録します。
- 'model-registry' プロジェクトに名前 'registered_model_name' の登録済みモデルが存在するかどうかを確認します。存在しない場合は、名前 'registered_model_name' の新しい登録済みモデルを作成します。
- モデル artifact 'name' のバージョンを登録済みモデル 'registered_model_name' にリンクします。
- 'aliases' リストからエイリアスを新しくリンクされたモデル artifact バージョンにアタッチします。

| Args |  |
| :--- | :--- |
|  `path` |  (str) このモデルのコンテンツへのパス。次の形式にすることができます。- `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `registered_model_name` |  (str) - モデルがリンクされる登録済みモデルの名前。登録済みモデルは、モデルレジストリにリンクされたモデルバージョンのコレクションであり、通常はチーム固有の ML タスクを表します。この登録済みモデルが属するエンティティーは、run から派生します。 |
|  `name` |  (str、オプション) - 'path' 内のファイルがログに記録されるモデル artifact の名前。指定されていない場合、パスの basename に現在の run ID が付加されたものがデフォルトになります。 |
|  `aliases` |  (List[str]、オプション) - このリンクされた artifact の登録済みモデル内でのみ適用されるエイリアス。エイリアス "latest" は、リンクされている artifact の最新バージョンに常に適用されます。 |

#### 例:

```python
run.link_model(
    path="/local/directory",
    registered_model_name="my_reg_model",
    name="my_model_artifact",
    aliases=["production"],
)
```

無効な使用法

```python
run.link_model(
    path="/local/directory",
    registered_model_name="my_entity/my_project/my_reg_model",
    name="my_model_artifact",
    aliases=["production"],
)

run.link_model(
    path="/local/directory",
    registered_model_name="my_reg_model",
    name="my_entity/my_project/my_model_artifact",
    aliases=["production"],
)
```

| Raises |  |
| :--- | :--- |
|  `AssertionError` |  registered_model_name がパスの場合、またはモデル artifact 'name' がサブストリング 'model' を含まないタイプの場合。 |
|  `ValueError` |  name に無効な特殊文字がある場合 |

| 戻り値 |  |
| :--- | :--- |
|  なし |

### `log`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1613-L1873)

```python
log(
    data: dict[str, Any],
    step: (int | None) = None,
    commit: (bool | None) = None,
    sync: (bool | None) = None
) -> None
```

run データをアップロードします。

`log` を使用して、スカラー、画像、ビデオ、
ヒストグラム、プロット、テーブルなどの run からデータをログに記録します。

ライブの例、コードスニペット、ベストプラクティスなどについては、
[ログ記録のガイド](https://docs.wandb.ai/guides/track/log) を参照してください。

最も基本的な使用法は `run.log({"train-loss": 0.5, "accuracy": 0.9})` です。
これにより、損失と精度が run の履歴に保存され、
これらのメトリクスの summary 値が更新されます。

ログに記録されたデータは、[wandb.ai](https://wandb.ai) のワークスペース、
または W&B アプリの [セルフホストインスタンス](https://docs.wandb.ai/guides/hosting) で、
またはデータをエクスポートして、ローカルで視覚化および探索できます。
たとえば、[API](https://docs.wandb.ai/guides/track/public-api-guide) を使用して、
Jupyter notebook で視覚化および探索できます。

ログに記録された値は、スカラーである必要はありません。wandb オブジェクトのログ記録はすべてサポートされています。
たとえば、`run.log({"example": wandb.Image("myimage.jpg")})` は、
W&B UI で適切に表示されるサンプル画像をログに記録します。
サポートされているさまざまなタイプについては、[リファレンスドキュメント](https://docs.wandb.com/ref/python/data-types) を参照するか、
例については、[ログ記録のガイド](https://docs.wandb.ai/guides/track/log) を確認してください。
3D 分子構造やセグメンテーションマスクから PR 曲線やヒストグラムまであります。
`wandb.Table` を使用して、構造化データをログに記録できます。詳細については、
[テーブルのログ記録のガイド](https://docs.wandb.ai/guides/models/tables/tables-walkthrough) を参照してください。

W&B UI は、名前のフォワードスラッシュ (`/`) を使用してメトリクスを整理し、
最後のスラッシュの前のテキストを使用してセクションに名前を付けます。たとえば、
次の結果は、"train" と "validate" という名前の 2 つのセクションになります。

```
run.log(
    {
        "train/accuracy": 0.9,
        "train/loss": 30,
        "validate/accuracy": 0.8,
        "validate/loss": 20,
    }
)
```

ネストのレベルは 1 つだけサポートされています。`run.log({"a/b/c": 1})`
は、"a/b" という名前のセクションを生成します。

`run.log` は、1 秒あたり数回以上呼び出されることを想定していません。
最適なパフォーマンスを得るには、ログ記録を N 回のイテレーションごとに 1 回に制限するか、
複数のイテレーションでデータを収集し、1 つのステップでログに記録します。

### W&B ステップ

基本的な使用法では、`log` の各呼び出しによって新しい "ステップ" が作成されます。
ステップは常に増加する必要があり、前のステップにログに記録することはできません。

グラフでは、任意のメトリクスを X 軸として使用できることに注意してください。
多くの場合、W&B ステップをトレーニングステップとしてではなく、
タイムスタンプとして扱う方が適しています。

```
# 例: X 軸として使用する "epoch" メトリクスをログに記録します。
run.log({"epoch": 40, "train-loss": 0.5})
```

[define_metric](https://docs.wandb.ai/ref/python/run#define_metric) も参照してください。

複数の `log` 呼び出しを使用して、
`step` および `commit` パラメータを使用して同じステップにログに記録することができます。
次はすべて同等です。

```
# 通常の使用法:
run.log({"train-loss": 0.5, "accuracy": 0.8})
run.log({"train-loss": 0.4, "accuracy": 0.9})

# 自動インクリメントなしの暗黙的なステップ:
run.log({"train-loss": 0.5}, commit=False)
run.log({"accuracy": 0.8})
run.log({"train-loss": 0.4}, commit=False)
run.log({"accuracy": 0.9})

# 明示的なステップ:
run.log({"train-loss": 0.5}, step=current_step)
run.log({"accuracy": 0.8}, step=current_step)
current_step += 1
run.log({"train-loss": 0.4}, step=current_step)
run.log({"accuracy": 0.9}, step=current_step)
```

| Args |  |
| :--- | :--- |
|  `data` |  `str` キーと、`int`、`float`、`string`、`wandb.data_types`、シリアル化可能な Python オブジェクトのリスト、タプル、NumPy 配列、この構造の他の `dict` など、シリアル化可能な Python オブジェクトである値を含む `dict`。 |
|  `step` |  ログに記録するステップ番号。`None` の場合、暗黙的な自動インクリメントステップが使用されます。説明のメモを参照してください。 |
|  `commit` |  true の場合、ステップを確定してアップロードします。false の場合、ステップのデータを累積します。説明のメモを参照してください。`step` が `None` の場合、デフォルトは `commit=True` です。それ以外の場合、デフォルトは `commit=False` です。 |
|  `sync` |  この引数は非推奨であり、何も行いません。 |

#### 例:

詳細な例については、
[ログ記録のガイド](https://docs.wandb.ai/guides/track/log) を参照してください。

### 基本的な使用法

```python
import wandb

run = wandb.init()
run.log({"accuracy": 0.9, "epoch": 5})
```

### 増分ログ記録

```python
import wandb

run = wandb.init()
run.log({"loss": 0.2}, commit=False)
# Somewhere else when I'm ready to report this step:
# 別の場所で、このステップを報告する準備ができたとき:
run.log({"accuracy": 0.8})
```

### ヒストグラム

```python
import numpy as np
import wandb

# sample gradients at random from normal distribution
# 正規分布からランダムに勾配をサンプリングします
gradients = np.random.randn(100, 100)
run = wandb.init()
run.log({"gradients": wandb.Histogram(gradients)})
```

### numpy からの画像

```python
import numpy as np
import wandb

run = wandb.init()
examples = []
for i in range(3):
    pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
    image = wandb.Image(pixels, caption=f"random field {i}")
    examples.append(image)
run.log({"examples": examples})
```

### PIL からの画像

```python
import numpy as np
from PIL import Image as PILImage
import wandb

run = wandb.init()
examples = []
for i in range(3):
    pixels = np.random.randint(
        low=0,
        high=256,
        size=(100, 100, 3),
        dtype=np.uint8,
    )
    pil_image = PILImage.fromarray(pixels, mode="RGB")
    image = wandb.Image(pil_image, caption=f"random field {i}")
    examples.append(image)
run.log({"examples": examples})
```

### numpy からのビデオ

```python
import numpy as np
import wandb

run = wandb.init()
# axes are (time, channel, height, width)
# 軸は (時間、チャンネル、高さ、幅) です
frames = np.random.randint(
    low=0,
    high=256,
    size=(10, 3, 100, 100),
    dtype=np.uint8,
)
run.log({"video": wandb.Video(frames, fps=4)})
```

### Matplotlib プロット

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

run = wandb.init()
fig, ax = plt.subplots()
x = np.linspace(0, 10)
y = x * x
ax.plot(x, y)  # plot y = x^2
# y = x^2 をプロットします
run.log({"chart": fig})
```

### PR 曲線

```python
import wandb

run = wandb.init()
run.log({"pr": wandb.plot.pr_curve(y_test, y_probas, labels)})
```

### 3D オブジェクト

```python
import wandb

run = wandb.init()
run.log(
    {
        "generated_samples": [
            wandb.Object3D(open("sample.obj")),
            wandb.Object3D(open("sample.gltf")),
            wandb.Object3D(open("sample.glb")),
        ]
    }
)
```

| Raises |  |
| :--- | :--- |
|  `wandb.Error` |  `wandb.init` の前に呼び出された場合 |
|  `ValueError` |  無効なデータが渡された場合 |

### `log_artifact`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3067-L3107)

```python
log_artifact(
    artifact_or_path: (Artifact | StrPath),
    name: (str | None) = None,
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    tags: (list[str] | None) = None
) -> Artifact
```

artifact を run の出力として宣言します。

| Args |  |
| :--- | :--- |
|  `artifact_or_path` |  (str または Artifact) この artifact のコンテンツへのパス。次の形式にすることができます。- `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` `wandb.Artifact` を呼び出して作成された Artifact オブジェクトを渡すこともできます。 |
|  `name` |  (str、オプション) artifact 名。有効な名前は、次の形式にすることができます。- name:version - name:alias - digest これが指定されていない場合、パスの basename に現在の run ID が付加されたものがデフォルトになります。 |
|  `type` |  (str) ログに記録する artifact のタイプ (例: `dataset`、`model`) |
|  `aliases` |  (list、オプション) この artifact に適用するエイリアス。デフォルトは `["latest"]` です。 |
|  `tags` |  (list、オプション) この artifact に適用するタグ (存在する場合)。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `log_code`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1004-L1089)

```python
log_code(
    root: (str | None) = ".",
    name: (str | None) = None,
    include_fn: (Callable[[str, str], bool] | Callable[[str], bool]) = _is_py_requirements_or_dockerfile,
    exclude_fn: (Callable[[str, str], bool] | Callable[[str], bool]) = filenames.exclude_wandb_fn
) -> (Artifact | None)
```

コードの現在の状態を W&B Artifact に保存します。

デフォルトでは、現在のディレクトリーを調べて、`.py` で終わるすべてのファイルをログに記録します。

| Args |  |
| :--- | :--- |
|  `root` |  コードを再帰的に検索する相対パス ( `os.getcwd()` ) または絶対パス。 |
|  `name` |  (str、オプション) コード artifact の名前。デフォルトでは、artifact に `source-$PROJECT_ID-$ENTRYPOINT_RELPATH` という名前を付けます。多くの run で同じ artifact を共有したい場合があります。name を指定すると、それを実現できます。 |
|  `include_fn` |  ファイルパスと (オプションで) ルートパスを受け入れ、含める場合は True、それ以外の場合は False を返す callable。これはデフォルトで次のようになります: `lambda path, root: path.endswith(".py")` |
|  `exclude_fn` |  ファイルパスと (オプションで) ルートパスを受け入れ、除外する場合は `True`、それ以外の場合は `False` を返す callable。これはデフォルトで、`<root>/.wandb/` および `<root>/wandb/` ディレクトリー内のすべてのファイルを除外する関数です。 |

#### 例:

基本的な使用法

```python
run.log_code()
```

高度な使用法

```python
run.log_code(
    "../",
    include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
    exclude_fn=lambda path, root: os.path.relpath(path, root).startswith(
        "cache/"
    ),
)
```

| 戻り値 |  |
| :--- | :--- |
|  コードがログに記録された場合は `Artifact` オブジェクト |

### `log_model`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3358-L3407)

```python
log_model(
    path: StrPath,
    name: (str | None) = None,
    aliases: (list[str] | None) = None
) -> None
```

'path' 内のコンテンツを含むモデル artifact を run にログに記録し、この run の出力としてマークします。

| Args |  |
| :--- | :--- |
|  `path` |  (str) このモデルのコンテンツへのパス。次の形式にすることができます。- `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `name` |  (str、オプション) ファイルコンテンツが追加されるモデル artifact に割り当てる名前。文字列には、次の英数字、ダッシュ、アンダースコア、ドットのみを含める必要があります。指定されていない場合、パスの basename に現在の run ID が付加されたものがデフォルトになります。 |
|  `aliases` |  (list、オプション) 作成されたモデル artifact に適用するエイリアス。デフォルトは `["latest"]` です。 |

#### 例:

```python
run.log_model(
    path="/local/directory",
    name="my_model_artifact",
    aliases=["production"],
)
```

無効な使用法

```python
run.log_model(
    path="/local/directory",
    name="my_entity/my_project/my_model_artifact",
    aliases=["production"],
)
```

| Raises |  |
| :--- | :--- |
|  `ValueError` |  name に無効な特殊文字がある場合 |

| 戻り値 |  |
| :--- | :--- |
|  なし |

### `mark_preempting`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3618-L3626)

```python
mark_preempting() -> None
```

この run をプリエンプトとしてマークします。

また、内部プロセスにこれをすぐにサーバーに報告するように指示します。

### `project_name`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L994-L996)

```python
project_name() -> str
```

### `restore`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2060-L2073)

```python
restore(
    name: str,
    run_path: (str | None) = None,
    replace: bool = (False),
    root: (str | None) = None
) -> (None | TextIO)
```

クラウドストレージから指定されたファイルをダウンロードします。

ファイルは、現在のディレクトリーまたは run ディレクトリーに配置されます。
デフォルトでは、ファイルがまだ存在しない場合にのみダウンロードされます。

| Args |  |
| :--- | :--- |
|  `name` |  ファイルの名前 |
|  `run_path` |  ファイルをプルする run へのオプションのパス。つまり、wandb.init が呼び出されていない場合は `username/project_name/run_id` が必要です。 |
|  `replace` |  ファイルがローカルに既に存在する場合でも、ファイルをダウンロードするかどうか |
|  `root` |  ファイルをダウンロードするディレクトリー。デフォルトは、現在のディレクトリー、または wandb.init が呼び出された場合は run ディレクトリーです。 |

| 戻り値 |  |
| :--- | :--- |
|  ファイルが見つからない場合は None。それ以外の場合は、読み取り用に開いているファイルオブジェクト |

| Raises |  |
| :--- | :--- |
|  `wandb.CommError` |  wandb バックエンドに接続できない場合 |
|  `ValueError` |  ファイルが見つからない場合、または run_path が見つからない場合 |

### `save`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1875-L1979)

```python
save(
    glob_str: (str | os.PathLike | None) = None,
    base_path: (str | os.PathLike | None) = None,
    policy: PolicyName = "live"
) -> (bool | list[str])
```

1 つまたは複数のファイルを W&B に同期します。

相対パスは、現在の作業ディレクトリーに対する相対パスです。

"myfiles/*" などの Unix glob は、`policy` に関係なく、`save` が
呼び出されたときに展開されます。特に、新しいファイルは
自動的に選択されません。

`base_path` を指定して、アップロードされたファイルのディレクトリー構造を
制御できます。これは `glob_str` のプレフィックスである必要があり、その下のディレクトリー
構造が保持されます。これは、例を通して理解するのが最適です。

```
wandb.save("these/are/myfiles/*")
# => Saves files in a "these/are/myfiles/" folder in the run.
# => run の "these/are/myfiles/" フォルダーにファイルを保存します。

wandb.save("these/are/myfiles/*", base_path="these")
# => Saves files in an "are/myfiles/" folder in the run.
# => run の "are/myfiles/" フォルダーにファイルを保存します。

wandb.save("/User/username/Documents/run123/*.txt")
# => Saves files in a "run123/" folder in the run. See note below.
# => run の "run123/" フォルダーにファイルを保存します。以下のメモを参照してください。

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => Saves files in a "username/Documents/run123/" folder in the run.
# => run の "username/Documents/run123/" フォルダーにファイルを保存します。

wandb.save("files/*/saveme.txt")
# => Saves each "saveme.txt" file in an appropriate subdirectory
# => 各 "saveme.txt" ファイルを適切なサブディレクトリーに保存します
#    of "files/".
#    "files/" の。
```

注: 絶対パスまたは glob が指定され、`base_path` が指定されていない場合、上記の例のように、1 つのディレクトリーレベルが保持されます。

| Args |  |
| :--- | :--- |
|  `glob_str` |  相対パスまたは絶対パス、あるいは Unix glob。 |
|  `base_path` |  ディレクトリー構造を推測するために使用するパス。例を参照してください。 |
|  `policy` |  `live`、`now`、または `end` のいずれか。* live: ファイルが変更されると、ファイルをアップロードし、以前のバージョンを上書きします * now: ファイルを今すぐ 1 回アップロードします * end: run が終了したときにファイルをアップロードします |

| 戻り値 |  |
| :--- | :--- |
|  一致したファイルに対して作成されたシンボリックリンクへのパス。履歴上の理由から、これはレガシーコードでブール値を返す場合があります。 |

### `status`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2161-L2183)

```python
status() -> RunStatus
```

現在の run の同期ステータスに関する、内部バックエンドからの同期情報を取得します。

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1238-L1247)

```python
to_html(
    height: int = 420,
    hidden: bool = (False)
) -> str
```

現在の run を表示する iframe を含む HTML を生成します。

### `unwatch`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2838-L2848)

```python
unwatch(
    models: (torch.nn.Module | Sequence[torch.nn.Module] | None) = None
) -> None
```

pytorch モデルのトポロジ、勾配、およびパラメータフックを削除します。

| Args |  |
| :--- | :--- |
|  models (torch.nn.Module | Sequence[torch.nn.Module]): ウォッチが呼び出された pytorch モデルのオプションのリスト |

### `upsert_artifact`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3109-L3161)

```python
upsert_artifact(
    artifact_or_path: (Artifact | str),
    name: (str | None) = None,
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    distributed_id: (str | None) = None
) -> Artifact
```

run の出力として、非 final な artifact を宣言します (または追加します)。

artifact を final にするには、run.finish_artifact() を呼び出す必要があることに注意してください。
これは、分散ジョブがすべて同じ artifact に貢献する必要がある場合に役立ちます。

| Args |  |
| :--- | :--- |
|  `artifact_or_path` |  (str または Artifact) この artifact のコンテンツへのパス。次の形式にすることができます。- `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` `wandb.Artifact` を呼び出して作成された Artifact オブジェクトを渡すこともできます。 |
|  `name` |  (str、オプション) artifact 名。エンティティー/プロジェクトでプレフィックスを付けることができます。有効な名前は、次の形式にすることができます。- name:version - name:alias - digest これが指定されていない場合、パスの basename に現在の run ID が付加されたものがデフォルトになります。 |
|  `type` |  (str) ログに記録する artifact のタイプ (例: `dataset`、`model`) |
|  `aliases` |  (list、オプション) この artifact に適用するエイリアス。デフォルトは `["latest"]` です。 |
|  `distributed_id` |  (string、オプション) すべての分散ジョブが共有する一意の文字列。None の場合、run のグループ名がデフォルトになります。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `use_artifact`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2953-L3065)

```python
use_artifact(
    artifact_or_name: (str | Artifact),
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    use_as: (str | None) = None
) -> Artifact
```

artifact を run への入力として宣言します。

返されたオブジェクトで `download` または `file` を呼び出して、コンテンツをローカルで取得します。

| Args |  |
| :--- | :--- |
|  `artifact_or_name` |  (str または Artifact) artifact 名。プロジェクト/またはエンティティー/プロジェクト/でプレフィックスを付けることができます。名前でエンティティーが指定されていない場合、Run または API 設定のエンティティーが使用されます。有効な名前は、次の形式にすることができます。- name:version - name:alias `wandb.Artifact` を呼び出して作成された Artifact オブジェクトを渡すこともできます。 |
|  `type` |  (str、オプション) 使用する artifact のタイプ。 |
|  `aliases` |  (list、オプション) この artifact に適用するエイリアス |
|  `use_as` |  (string、オプション) artifact が使用された目的を示すオプションの文字列。UI に表示されます。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `use_model`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3409-L3464)

```python
use_model(
    name: str
) -> FilePathStr
```

モデル artifact 'name' にログに記録されたファイルをダウンロードします。

| Args |  |
| :--- | :--- |
|  `name` |  (str) モデル artifact 名。'name' は、既存のログに記録されたモデル artifact の名前と一致する必要があります。エンティティー/プロジェクト/でプレフィックスを付けることができます。有効な名前は、次の形式にすることができます。- model_artifact_name:version - model_artifact_name:alias |

#### 例:

```python
run.use_model(
    name="my_model_artifact:latest",
)

run.use_model(
    name="my_project/my_model_artifact:v0",
)

run.use_model(
    name="my_entity/my_project/my_model_artifact:<digest>",
)
```

無効な使用法

```python
run.use_model(
    name="my_entity/my_project/my_model_artifact",
)
```

| Raises |  |
| :--- | :--- |
|  `AssertionError` |  モデル artifact 'name' が、サブストリング 'model' を含まないタイプの場合。 |

| 戻り値 |  |
| :--- | :--- |
|  `path` |  (str) ダウンロードされたモデル artifact ファイルへのパス。 |

### `watch`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2801-L2836)

```python
watch(
    models: (torch.nn.Module | Sequence[torch.nn.Module]),
    criterion: (torch.F | None) = None,
    log: (Literal['gradients', 'parameters', 'all'] | None) = "gradients",
    log_freq: int = 1000,
    idx: (int | None) = None,
    log_graph: bool = (False)
) -> None
```

指定された PyTorch モデルにフックして、勾配とモデルの計算グラフを監視します。

この関数は、トレーニング中にパラメータ、勾配、またはその両方を追跡できます。将来は、任意の 機械学習 モデルをサポートするように拡張する必要があります。

| Args |  |
| :--- | :--- |
|  models (Union[torch.nn.Module, Sequence[torch.nn.Module]]): 監視する単一のモデルまたはモデルのシーケンス。criterion (Optional[torch.F]): 最適化される損失関数 (オプション)。log (Optional[Literal["gradients", "parameters", "all"]]): "gradients"、"parameters"、または "all" のいずれをログに記録するかを指定します。ログ記録を無効にするには None に設定します。(デフォルト="gradients") log_freq (int): 勾配とパラメータをログに記録する頻度 (バッチ単位)。(デフォルト=1000) idx (Optional[int]): `wandb.watch` で複数のモデルを追跡する場合に使用されるインデックス。(デフォルト=None) log_graph (bool): モデルの計算グラフをログに記録するかどうか。(デフォルト=False) |

| Raises |  |
| :--- | :--- |
|  `ValueError` |  `wandb.init` が呼び出されていない場合、またはモデルのいずれかが `torch.nn.Module` のインスタンスでない場合。 |

### `__enter__`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3602-L3603)

```python
__enter__() -> Run
```

### `__exit__`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3605-L3616)

```python
__exit__(
    exc_type: type[BaseException],
    exc_val: BaseException,
    exc_tb: TracebackType
) -> bool
```
