---
title: run
menu:
  reference:
    identifier: ja-ref-python-run
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L461-L4042 >}}

wandbによってログに記録される計算の単位。通常、これは機械学習実験です。

```python
Run(
    settings: Settings,
    config: (dict[str, Any] | None) = None,
    sweep_config: (dict[str, Any] | None) = None,
    launch_config: (dict[str, Any] | None) = None
) -> None
```

`wandb.init()`を使用してrunを作成します：

```python
import wandb

run = wandb.init()
```

どのプロセスにも最大で1つだけアクティブな`wandb.Run`があり、`wandb.run`としてアクセス可能です：

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```

`wandb.log`でログに記録したものはすべてそのrunに送信されます。

同じスクリプトまたはノートブックで複数のrunを開始したい場合は、進行中のrunを終了する必要があります。Runは`wandb.finish`で終了するか、`with`ブロック内で使用することで終了できます：

```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
    pass  # データをここでログに記録

assert wandb.run is None
```

`wandb.init`でrunを作成する方法についてはドキュメントを参照するか、
[こちらのガイド](https://docs.wandb.ai/guides/track/launch)をご覧ください。

分散トレーニングでは、ランク0のプロセスで単一のrunを作成してそのプロセスからのみ情報をログするか、各プロセスでrunを作成してそれぞれからログを取り、`wandb.init`の`group`引数で結果をグループ化することができます。W&Bを使用した分散トレーニングの詳細については、
[こちらのガイド](https://docs.wandb.ai/guides/track/log/distributed-training)をご覧ください。

現在、`wandb.Api`には並行する`Run`オブジェクトがあります。最終的にこれら2つのオブジェクトは統合される予定です。

| 属性 |  |
| :--- | :--- |
|  `summary` |  (Summary) 各`wandb.log()`キーに設定された単一の値。デフォルトでは、summaryは最後にログした値に設定されます。summaryを手動で最高の値（例: 最大精度）に設定することもできます。 |
|  `config` |  このrunに関連付けられたConfigオブジェクト。 |
|  `dir` |  runに関連するファイルが保存されるディレクトリ。 |
|  `entity` |  runに関連するW&Bエンティティの名前。エンティティはユーザー名、チーム名、または組織名です。 |
|  `group` |  runに関連するグループ名。グループを設定すると、W&B UIがrunを整理しやすくなります。分散トレーニングをしている場合、トレーニング内のすべてのrunに同じグループを与える必要があります。クロスバリデーションをしている場合、すべてのクロスバリデーションフォールドに同じグループを与える必要があります。 |
|  `id` |  このrunの識別子。 |
|  `mode` |  `0.9.x`およびそれ以前との互換性のためのもので、最終的には非推奨になります。 |
|  `name` |  runの表示名。表示名は一意であることを保証されず、説明的である可能性があります。デフォルトでは、ランダムに生成されます。 |
|  `notes` |  runに関連付けられたノートがあれば表示されます。ノートは複数行の文字列で、マークダウンやlatex方程式を`$$`の中で使用できます（例: `$x + 3$`）。 |
|  `path` |  runへのパス。Runパスには、`entity/project/run_id`の形式でエンティティ、プロジェクト、およびrun IDが含まれます。 |
|  `project` |  runに関連するW&Bプロジェクトの名前。 |
|  `resumed` |  runが再開された場合はTrue、それ以外はFalse。 |
|  `settings` |  runの設定オブジェクトの凍結コピー。 |
|  `start_time` |  runが開始されたUnixタイムスタンプ（秒）。 |
|  `starting_step` |  runの最初のステップ。 |
|  `step` |  現在のステップの値。このカウンターは`wandb.log`によってインクリメントされます。 |
|  `sweep_id` |  ある場合はrunに関連するsweepのID。 |
|  `tags` |  runに関連するタグがあれば表示されます。 |
|  `url` |  runに関連するW&Bのurl。 |

## メソッド

### `alert`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3567-L3600)

```python
alert(
    title: str,
    text: str,
    level: (str | AlertLevel | None) = None,
    wait_duration: (int | float | timedelta | None) = None
) -> None
```

指定されたタイトルとテキストでアラートを開始します。

| 引数 |  |
| :--- | :--- |
|  `title` |  (str) アラートのタイトル。64文字未満である必要があります。 |
|  `text` |  (str) アラートの本文。 |
|  `level` |  (strまたはAlertLevel、オプショナル) 使用するアラートレベル。`INFO`、`WARN`、または`ERROR`のいずれかです。 |
|  `wait_duration` |  (int, float, またはtimedelta、オプショナル) このタイトルで別のアラートを送信する前に待つ時間（秒）。 |

### `define_metric`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2660-L2721)

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

`wandb.log()`で記録されたメトリクスをカスタマイズします。

| 引数 |  |
| :--- | :--- |
|  `name` |  カスタマイズするメトリクスの名前。 |
|  `step_metric` |  このメトリクスのX軸として機能する他のメトリクスの名前。 |
|  `step_sync` |  明示的に提供されない場合、最後のstep_metricの値を`run.log()`に自動で挿入します。step_metricが指定されている場合、デフォルトはTrueです。 |
|  `hidden` |  このメトリクスを自動プロットから非表示にします。 |
|  `summary` |  summaryに追加される集計メトリクスを指定します。サポートされている集計には「min」、「max」、「mean」、「last」、「best」、「copy」、「none」が含まれます。「best」はgoalパラメータと共に使用します。「none」はsummaryの生成を防ぎます。「copy」は非推奨で使用しないでください。 |
|  `goal` |  "best" summaryタイプの解釈方法を指定します。サポートされているオプションは「minimize」と「maximize」です。 |
|  `overwrite` |  Falseの場合、同じメトリクスのために以前の`define_metric`呼び出しとこの呼び出しがマージされ、指定されていないパラメータには以前の呼び出しで指定された値が使用されます。Trueの場合、指定されていないパラメータは以前の呼び出しで指定された値を上書きします。 |

| 戻り値 |  |
| :--- | :--- |
|  この呼び出しを表すオブジェクトですが、他には捨てても問題ありません。 |

### `detach`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2885-L2886)

```python
detach() -> None
```

### `display`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1219-L1236)

```python
display(
    height: int = 420,
    hidden: bool = (False)
) -> bool
```

このrunをjupyterで表示します。

### `finish`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2075-L2106)

```python
finish(
    exit_code: (int | None) = None,
    quiet: (bool | None) = None
) -> None
```

runを終了し、残りのデータをアップロードします。

W&B runの完了をマークし、すべてのデータがサーバーに同期されていることを確認します。
runの最終状態は、その終了条件と同期状態によって決まります。

#### Runの状態:

- Running: データをログしているおよび/またはハートビートを送信しているアクティブなrun。
- Crashed: 予期せずハートビートの送信を停止したrun。
- Finished: すべてのデータが同期されて正常に完了したrun（`exit_code=0`）。
- Failed: エラーで完了したrun（`exit_code!=0`）。

| 引数 |  |
| :--- | :--- |
|  `exit_code` |  runの終了ステータスを示す整数。成功の場合は0、他の値はrunを失敗としてマークします。 |
|  `quiet` |  廃止予定。`wandb.Settings(quiet=...)`を使用してログの冗長性を設定します。 |

### `finish_artifact`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3163-L3215)

```python
finish_artifact(
    artifact_or_path: (Artifact | str),
    name: (str | None) = None,
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    distributed_id: (str | None) = None
) -> Artifact
```

非最終アーティファクトをrunの出力として終了します。

同じdistributed IDでの後続の「アップサート」は新しいバージョンになります。

| 引数 |  |
| :--- | :--- |
|  `artifact_or_path` |  (strまたはArtifact） このアーティファクトの内容へのパス、次の形式で可能： - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` また、`wandb.Artifact`を呼び出すことによって作成されたArtifactオブジェクトを渡すこともできます。 |
|  `name` |  (str, オプション) アーティファクト名。entity/projectで接頭辞を付ける場合もあります。次の形式で有効な名前にできます： - name:version - name:alias - digest 指定されていない場合、デフォルトでパスのベース名に現在のidが追加されます。 |
|  `type` |  (str) ログを記録するアーティファクトのタイプ、例：`dataset`、`model` |
|  `aliases` |  (list, オプション) このアーティファクトに適用するエイリアス。デフォルトは`["latest"]`。 |
|  `distributed_id` |  (文字列, オプション) すべての分散ジョブが共有する一意の文字列。Noneの場合、runのグループ名がデフォルトです。 |

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトオブジェクト。 |

### `get_project_url`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1091-L1099)

```python
get_project_url() -> (str | None)
```

runに関連付けられたW&BプロジェクトのURLを返します（存在する場合）。

オフラインrunはプロジェクトURLを持ちません。

### `get_sweep_url`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1101-L1106)

```python
get_sweep_url() -> (str | None)
```

runに関連付けられたsweepのURLを返します（存在する場合）。

### `get_url`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1108-L1116)

```python
get_url() -> (str | None)
```

W&B runのURLを返します（存在する場合）。

オフラインrunはURLを持ちません。

### `join`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2148-L2159)

```python
join(
    exit_code: (int | None) = None
) -> None
```

`finish()`のための非推奨のエイリアス - 代わりにfinishを使用してください。

### `link_artifact`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2888-L2951)

```python
link_artifact(
    artifact: Artifact,
    target_path: str,
    aliases: (list[str] | None) = None
) -> None
```

指定されたアートファクトをポートフォリオ（アーティストの昇格されたコレクション）にリンクします。

リンクされたアートファクトは、指定されたポートフォリオのUIに表示されます。

| 引数 |  |
| :--- | :--- |
|  `artifact` |  公開またはローカルアートファクトで、リンクされるアーティファクト。 |
|  `target_path` |  `str` - 次の形式を取り得る： `{portfolio}`, `{project}/{portfolio}`, または `{entity}/{project}/{portfolio}` |
|  `aliases` |  `List[str]` - このリンクアーティファクト内のポートフォリオでのみ適用されるオプショナルなエイリアス。 "latest"のエイリアスは、リンクされたアーティファクトの最新バージョンに常に適用されます。 |

| 戻り値 |  |
| :--- | :--- |
|  None |

### `link_model`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3466-L3565)

```python
link_model(
    path: StrPath,
    registered_model_name: str,
    name: (str | None) = None,
    aliases: (list[str] | None) = None
) -> None
```

モデルアーティファクトバージョンをログし、モデルレジストリで登録されたモデルにリンクします。

リンクされたモデルバージョンは、指定された登録モデルのUIに表示されます。

#### ステップ:

- 'name'モデルアーティファクトがログされているか確認します。そうであれば、'path'にあるファイルに一致するアーティファクトバージョンを使用するか新しいバージョンをログします。そうでなければ、'path'の下のファイルを新しいモデルアーティファクト'type'の'type'としてログします。
- 'model-registry'プロジェクトに'registered_model_name'という名前の登録モデルが存在するか確認します。存在しない場合、'registered_model_name'という名前の新しい登録モデルを作成します。
- モデルアーティファクト'name'のバージョンを登録モデル'registered_model_name'にリンクします。
- 新しくリンクされたモデルアーティファクトバージョンに'aliases'リストのエイリアスを添付します。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) モデルの内容へのパスは、次の形式で可能です： - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `registered_model_name` |  (str) - モデルがリンクされる登録モデルの名前。登録モデルはモデルバージョンのコレクションであり、通常はチームの特定のMLタスクを表します。この登録モデルが属するエンティティはrunから派生します。 |
|  `name` |  (str, オプション) - 'path'のファイルがログされるモデルアーティファクトの名前です。指定されていない場合、デフォルトでパスのベース名に現在のrun idが付加されます。 |
|  `aliases` |  (List[str], オプション) - このリンクされたアーティファクト内の登録モデルにのみ適用されるエイリアス。リンクされたアーティファクトの最新バージョンには常に"latest"のエイリアスが適用されます。 |

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

| 発生する可能性のあるエラー |  |
| :--- | :--- |
|  `AssertionError` |  registered_model_nameがパスであるか、モデルアーティファクト名が「model」の部分文字列を持っていない場合 |
|  `ValueError` |  nameに無効な特殊文字が含まれている場合 |

| 戻り値 |  |
| :--- | :--- |
|  None |

### `log`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1613-L1873)

```python
log(
    data: dict[str, Any],
    step: (int | None) = None,
    commit: (bool | None) = None,
    sync: (bool | None) = None
) -> None
```

runデータをアップロードします。

`log`を使用してrunからデータ（例えば、スカラー、画像、ビデオ、ヒストグラム、プロット、テーブルなど）をログに記録します。

[ログ記録ガイド](https://docs.wandb.ai/guides/track/log)を参照し、
ライブの例、コードスニペット、ベストプラクティスなどを確認してください。

最も基本的な使用法は`run.log({"train-loss": 0.5, "accuracy": 0.9})`です。
これは、損失と精度をrunの履歴に保存し、これらのメトリクスのsummaryの値を更新します。

ワークスペースでログしたデータを[wandb.ai](https://wandb.ai)で可視化したり、自分でホストするW&Bアプリのインスタンスでローカルに可視化したり、データをエクスポートしてローカルで可視化、探索したり、例えば、Jupyter ノートブックで[公開API](https://docs.wandb.ai/guides/track/public-api-guide)を使用して行うことができます。

ログした値はスカラーである必要はありません。任意のwandbオブジェクトのログがサポートされています。
例えば、`run.log({"example": wandb.Image("myimage.jpg")})`は、W&B UIにうまく表示される例の画像のログを記録します。
さまざまにサポートされているタイプについては、[参照ドキュメント](https://docs.wandb.ai/ref/python/sdk/data-types/)を参照するか、[logging ガイド](https://docs.wandb.ai/guides/track/log)で、3D分子構造やセグメンテーションマスク、PR曲線、ヒストグラムの例を確認してください。
`wandb.Table`を使用して構造化データをログできます。詳細については、[テーブルログのガイド](https://docs.wandb.ai/guides/models/tables/tables-walkthrough)をご覧ください。

W&B UIは、メトリクスを名前に含むスラッシュ（`/`）を含むセクションに整理し、名前の最後のスラッシュ前のテキストでセクション名を使用します。例えば、次の例では、「train」と「validate」という2つのセクションがあります：

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

サポートされているネストのレベルは1つのみです。`run.log({"a/b/c": 1})`は「a/b」という名前のセクションを生成します。

`run.log`は1秒に数回以上呼び出す目的には使用されていません。
最適なパフォーマンスのために、N回のイテレーションごとに一度ログを記録するか、複数のイテレーションにわたってデータを収集し、単一のステップでログを記録することをお勧めします。

### W&Bステップ

基本的な使用法では、`log`を呼び出すたびに新しい「ステップ」が作成されます。
ステップは常に増加する必要があり、過去のステップにログを記録することはできません。

多くの場合、W&Bステップをタイムスタンプと同様に扱う方がトレーニングステップとして扱うより良いです。

```
# 例: X軸として使用する "epoch" メトリクスをログする方法。
run.log({"epoch": 40, "train-loss": 0.5})
```

[define_metric](https://docs.wandb.ai/ref/python/sdk/classes/run/#method-rundefine_metric)も参照してください。

`step`および`commit`パラメータを使用して、同じステップにログを記録するために複数の`log`呼び出しを使用することができます。
以下はすべて同等です：

```
# 通常の使用法：
run.log({"train-loss": 0.5, "accuracy": 0.8})
run.log({"train-loss": 0.4, "accuracy": 0.9})

# 自動インクリメントなしの暗黙のステップ：
run.log({"train-loss": 0.5}, commit=False)
run.log({"accuracy": 0.8})
run.log({"train-loss": 0.4}, commit=False)
run.log({"accuracy": 0.9})

# 明示的なステップ：
run.log({"train-loss": 0.5}, step=current_step)
run.log({"accuracy": 0.8}, step=current_step)
current_step += 1
run.log({"train-loss": 0.4}, step=current_step)
run.log({"accuracy": 0.9}, step=current_step)
```

| 引数 |  |
| :--- | :--- |
|  `data` |  シリアル化可能なPythonオブジェクトを含む`int`、`float`、および`string`、`wandb.data_types`のいずれか；シリアル化可能なPythonオブジェクトのリスト、タプルとNumPy配列。;この構造の他の`dict`。 |
|  `step` |  ログするステップ番号。もし`None`の場合、暗黙の自動インクリメントステップが使用されます。詳細は説明の注意事項を参照してください。 |
|  `commit` |  真実である場合、ステップが完了してアップロードされます。偽の場合、データはステップのために蓄積されます。詳細は説明の注意事項を参照してください。ステップが`None`の場合、デフォルトは`commit=True`のままで、それ以外の場合にはデフォルトは`commit=False`です。 |
|  `sync` |  この引数は廃止されており、何も行わない。 |

#### 例:

詳細でより多くの例は、
[このログ記録ガイド](https://docs.wandb.com/guides/track/log)をご覧ください。

### 基本的な使用法

```python
import wandb

run = wandb.init()
run.log({"accuracy": 0.9, "epoch": 5})
```

### インクリメンタルログ

```python
import wandb

run = wandb.init()
run.log({"loss": 0.2}, commit=False)
# 別の場所でこのステップを報告する準備ができたとき：
run.log({"accuracy": 0.8})
```

### ヒストグラム

```python
import numpy as np
import wandb

# グラデーションをランダムに正規分布からサンプリング
gradients = np.random.randn(100, 100)
run = wandb.init()
run.log({"gradients": wandb.Histogram(gradients)})
```

### NumPyから画像

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

### PILから画像

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

### NumPyからビデオ

```python
import numpy as np
import wandb

run = wandb.init()
# 軸 (time, channel, height, width)
frames = np.random.randint(
    low=0,
    high=256,
    size=(10, 3, 100, 100),
    dtype=np.uint8,
)
run.log({"video": wandb.Video(frames, fps=4)})
```

### Matplotlibプロット

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

run = wandb.init()
fig, ax = plt.subplots()
x = np.linspace(0, 10)
y = x * x
ax.plot(x, y)  # プロットy = x^2
run.log({"chart": fig})
```

### PR曲線

```python
import wandb

run = wandb.init()
run.log({"pr": wandb.plot.pr_curve(y_test, y_probas, labels)})
```

### 3Dオブジェクト

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

| 発生する可能性のあるエラー |  |
| :--- | :--- |
|  `wandb.Error` |  `wandb.init`を呼び出す前に呼び出された場合 |
|  `ValueError` |  無効なデータが渡された場合 |

### `log_artifact`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3067-L3107)

```python
log_artifact(
    artifact_or_path: (Artifact | StrPath),
    name: (str | None) = None,
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    tags: (list[str] | None) = None
) -> Artifact
```

アーティファクトをrunの出力として宣言します。

| 引数 |  |
| :--- | :--- |
|  `artifact_or_path` |  (strまたはArtifact） このアーティファクトの内容へのパス、次の形式で可能： - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` また、`wandb.Artifact`を呼び出すことによって作成されたArtifactオブジェクトを渡すこともできます。 |
|  `name` |  (str, オプション) アーティファクト名。次の形式で有効な名前にすることができます： - name:version - name:alias - digest 指定されていない場合、デフォルトでパスのベース名に現在のrun idが追加されます。 |
|  `type` |  (str) ログを記録するアーティファクトのタイプ、例：`dataset`、`model` |
|  `aliases` |  (list, オプション) このアーティファクトに適用するエイリアス。デフォルトは`["latest"]`。 |
|  `tags` |  (list, オプション) このアーティファクトに適用するタグ。もし存在すれば。 |

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトオブジェクト。 |

### `log_code`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1004-L1089)

```python
log_code(
    root: (str | None) = ".",
    name: (str | None) = None,
    include_fn: (Callable[[str, str], bool] | Callable[[str], bool]) = _is_py_requirements_or_dockerfile,
    exclude_fn: (Callable[[str, str], bool] | Callable[[str], bool]) = filenames.exclude_wandb_fn
) -> (Artifact | None)
```

コードの現状をW&Bアーティファクトとして保存します。

デフォルトでは、現在のディレクトリを探索し、`.py`で終わるすべてのファイルをログに記録します。

| 引数 |  |
| :--- | :--- |
|  `root` |  `os.getcwd()`に対して相対的、または絶対パスで、コードを再帰的に見つけるためのルート。 |
|  `name` |  (str, オプション) 私たちのコードアーティファクトの名前。デフォルトでは、アーティファクトは`source-$PROJECT_ID-$ENTRYPOINT_RELPATH`と名付けられます。シナリオによっては、複数のrunが同じアーティファクトを共有したい場合があります。nameを指定することでそれを達成できます。 |
|  `include_fn` |  ファイルパスと（オプションで）ルートパスを引数とし、含めるべき場合はTrueを返して、そうでない場合はFalseを返すcallable。デフォルトは：`lambda path, root: path.endswith(".py")` |
|  `exclude_fn` |  ファイルパスと（オプションで）ルートパスを引数とし、除外するべき場合はTrueを返して、そうでない場合はFalseを返すcallable。このデフォルトは、`<root>/.wandb/`および`<root>/wandb/`ディレクトリ内のすべてのファイルを除外する関数です。 |

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
|  コードがログされた場合のアーティファクトオブジェクト |

### `log_model`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3358-L3407)

```python
log_model(
    path: StrPath,
    name: (str | None) = None,
    aliases: (list[str] | None) = None
) -> None
```

'path'内の内容をrunに含むモデルアーティファクトをログに記録し、runの出力としてマークします。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) モデルの内容へのパスは、次の形式で可能です： - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `name` |  (str, オプション） ファイル内容が追加されるモデルアーティファクトに割り当てられる名前。文字列には、次のアルファベットと数字の文字のみ含めることができます：ダッシュ、アンダースコア、およびドット。指定されていない場合、デフォルトでパスのベース名に現在のrun idが追加されます。 |
|  `aliases` |  (list, オプション） 作成されたモデルアーティファクトに適用するエイリアスのデフォルトは「latest」です。 |

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

| 発生する可能性のあるエラー |  |
| :--- | :--- |
|  `ValueError` |  nameに無効な特殊文字が含まれている場合 |

| 戻り値 |  |
| :--- | :--- |
|  None |

### `mark_preempting`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3618-L3626)

```python
mark_preempting() -> None
```

このrunを中断しているとマークします。

また、内部プロセスに即座にサーバーに報告するよう指示します。

### `project_name`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L994-L996)

```python
project_name() -> str
```

### `restore`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2060-L2073)

```python
restore(
    name: str,
    run_path: (str | None) = None,
    replace: bool = (False),
    root: (str | None) = None
) -> (None | TextIO)
```

クラウドストレージから指定されたファイルをダウンロードします。

ファイルは現在のディレクトリまたはrunディレクトリに配置されます。
デフォルトでは、ローカルにまだ存在しない場合にのみファイルをダウンロードします。

| 引数 |  |
| :--- | :--- |
|  `name` |  ファイルの名前 |
|  `run_path` |  ファイルをプルするrunへのオプションのパス、例：`username/project_name/run_id`。wandb.initが呼び出されていない場合、これは必須です。 |
|  `replace` |  ファイルがローカルに既に存在する場合でもダウンロードするかどうか |
|  `root` |  ファイルをダウンロードするディレクトリ。デフォルトは現在のディレクトリまたはwandb.initが呼び出された場合のrunディレクトリ。 |

| 戻り値 |  |
| :--- | :--- |
|  ファイルが見つからない場合はNone、見つかった場合は読み取り用に開かれたファイルオブジェクト。 |

| 発生する可能性のあるエラー |  |
| :--- | :--- |
|  `wandb.CommError` |  wandbバックエンドに接続できない場合 |
|  `ValueError` |  ファイルが見つからない場合、またはrun_pathが見つからない場合 |

### `save`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1875-L1979)

```python
save(
    glob_str: (str | os.PathLike | None) = None,
    base_path: (str | os.PathLike | None) = None,
    policy: PolicyName = "live"
) -> (bool | list[str])
```

1つ以上のファイルをW&Bに同期させます。

相対的なパスは現在の作業ディレクトリに相対的です。

`save`が呼び出された時点でUnix glob（例えば「myfiles/*」）が展開され、`policy`に関係なく新しいファイルは自動的にピックアップされることはありません。

`save`を呼び出した時点でUnix glob（例えば"myfiles/*"）は、`policy`に関係なく展開されます。特に、新しいファイルは自動的に取得されません。

アップロードされるファイルのディレクトリ構造を制御するために`base_path`を指定できます。これは`glob_str`の接頭辞であり、その下のディレクトリ構造は保持されます。例を通じて最適に理解できます：

```
wandb.save("these/are/myfiles/*")
# => run内の"these/are/myfiles/"フォルダにファイルを保存します。

wandb.save("these/are/myfiles/*", base_path="these")
# => run内の"are/myfiles/"フォルダにファイルを保存します。

wandb.save("/User/username/Documents/run123/*.txt")
# => run内の"run123/"フォルダにファイルを保存します。以下の注意を参照してください。

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => run内の"username/Documents/run123/"フォルダにファイルを保存します。

wandb.save("files/*/saveme.txt")
# => 各 "saveme.txt" ファイルを "files/" の適切なサブディレクトリに保存します。
```

注意：絶対パスまたはglobが指定され`base_path`がない場合は、上記の例のように1つのディレクトリレベルが保持されます。

| 引数 |  |
| :--- | :--- |
|  `glob_str` |  相対または絶対のパスまたはUnix glob。 |
|  `base_path` |  ディレクトリ構造を推定するためのパス；例を参照。 |
|  `policy` |  `live`、`now`、または`end`のいずれか。 * live: ファイルを変更すると新しいバージョンで上書きしてアップロードします。 * now: 現在アップロードされているファイルを一度だけアップロードします。 * end: runが終了したときにファイルをアップロードします。 |

| 戻り値 |  |
| :--- | :--- |
|  発生した場合、ファイルのリンクシステムとして作成されたパス。歴史的な理由から、これが古いコードでブール値を返す場合もあります。 |

### `status`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2161-L2183)

```python
status() -> RunStatus
```

内部バックエンドからの現在のrunの同期状態についての情報を取得します。

### `to_html`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1238-L1247)

```python
to_html(
    height: int = 420,
    hidden: bool = (False)
) -> str
```

現在のrunを表示するiframeを含むHTMLを生成します。

### `unwatch`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2838-L2848)

```python
unwatch(
    models: (torch.nn.Module | Sequence[torch.nn.Module] | None) = None
) -> None
```

PyTorchモデルのトポロジー、勾配、およびパラメータフックを削除します。

| 引数 |  |
| :--- | :--- |
|  models (torch.nn.Module | Sequence[torch.nn.Module]): 呼び出し済みのwatchがあり、そのリストであるオプションのPyTorchモデル。 |

### `upsert_artifact`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3109-L3161)

```python
upsert_artifact(
    artifact_or_path: (Artifact | str),
    name: (str | None) = None,
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    distributed_id: (str | None) = None
) -> Artifact
```

runの出力として、非最終化されたアーティファクトを宣言する（または追加）します。

注意：アーティファクトを最終化するためには、 `run.finish_artifact()`を呼び出す必要があります。
これは分散ジョブが同じアーティファクトにすべて貢献する必要がある場合に役立ちます。

| 引数 |  |
| :--- | :--- |
|  `artifact_or_path` |  (strまたはArtifact） このアーティファクトの内容へのパス、次の形式で可能： - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` また、`wandb.Artifact`を呼び出すことによって作成されたArtifactオブジェクトを渡すこともできます。 |
|  `name` |  (str, オプション) アーティファクト名。entity/projectで接頭辞を付ける場合もあります。次の形式で有効な名前にできます： - name:version - name:alias - digest 指定されていない場合、デフォルトでパスのベース名に現在のrun idが追加されます。 |
|  `type` |  (str) ログを記録するアーティファクトのタイプ、例：`dataset`、`model` |
|  `aliases` |  (list, オプション) このアーティファクトに適用するエイリアス。デフォルトは`["latest"]`。 |
|  `distributed_id` |  (文字列, オプション) すべての分散ジョブが共有する一意の文字列。Noneの場合、runのグループ名がデフォルトです。 |

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトオブジェクト。 |

### `use_artifact`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2953-L3065)

```python
use_artifact(
    artifact_or_name: (str | Artifact),
    type: (str | None) = None,
    aliases: (list[str] | None) = None,
    use_as: (str | None) = None
) -> Artifact
```

runの入力としてアーティファクトを宣言します。

返されたオブジェクトで`download`または`file`を呼び出して、コンテンツをローカルに取得します。

| 引数 |  |
| :--- | :--- |
|  `artifact_or_name` |  (strまたはArtifact） アーティファクト名。project/またはentity/project/で接頭辞を付ける場合もあります。名前にentityが指定されていない場合、RunまたはAPIのエンティティ設定が使用されます。次の形式で有効な名前にできます： - name:version - name:alias または、`wandb.Artifact`の呼び出しで作成されたArtifactオブジェクトを渡すこともできます。 |
|  `type` |  (str, オプション) 使用するアーティファクトのタイプ。 |
|  `aliases` |  (list, オプション) このアーティファクトに適用するエイリアス。 |
|  `use_as` |  (文字列, オプション) オプショナル文字列で、そのアーティファクトがどんな目的で使用されたかを示します。UIで表示されます。 |

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトオブジェクト。 |

### `use_model`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3409-L3464)

```python
use_model(
    name: str
) -> FilePathStr
```

モデルアーティファクト'name'にログされたファイルをダウンロードします。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str） モデルアーティファクト名。'name'は既存のログされたモデルアーティファクトの名前と一致する必要があります。entity/project/で接頭辞を付ける場合もあります。次の形式で有効な名前にできます： - model_artifact_name:version - model_artifact_name:alias |

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

| 発生する可能性のあるエラー |  |
| :--- | :--- |
|  `AssertionError` |  モデルアーティファクト'name'が「model」の部分文字列を含むタイプではない場合。 |

| 戻り値 |  |
| :--- | :--- |
|  `path` |  (str） ダウンロードされたモデルアーティファクトファイルのパス。 |

### `watch`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2801-L2836)

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

与えられたPyTorchのモデルにフックを設定して、勾配とモデルの計算グラフを監視します。

この関数は、トレーニング中にパラメータ、勾配、または両方を追跡することができます。
将来的には任意の機械学習モデルに対応するよう拡張されるべきです。

| 引数 |  |
| :--- | :--- |
|  models (Union[torch.nn.Module, Sequence[torch.nn.Module]]): モニタリングされるモデルまたはモデルのシーケンス。 criterion (Optional[torch.F]): 最適化される損失関数 (オプション)。 log (Optional[Literal["gradients", "parameters", "all"]]): "gradients"、"parameters"、または"all" をどれをログするか指定します。ログを無効にするにはNoneを設定します (デフォルトは"gradients")。 log_freq (int): 勾配とパラメータをログする頻度 (バッチ単位）。 (デフォルトは1000）。 idx (Optional[int]): `wandb.watch`を使用して複数のモデルを追跡する時に使用されるインデックス (デフォルトはNone）。 log_graph (bool): モデルの計算グラフをログするかどうか。 (デフォルトはFalse） |

| 発生する可能性のあるエラー |  |
| :--- | :--- |
|  `ValueError` |  `wandb.init`が呼び出されていない場合や、モデルのいずれかが`torch.nn.Module`のインスタンスでない場合。 |

### `__enter__`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3602-L3603)

```python
__enter__() -> Run
```

### `__exit__`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L3605-L3616)

```python
__exit__(
    exc_type: type[BaseException],
    exc_val: BaseException,
    exc_tb: TracebackType
) -> bool
```