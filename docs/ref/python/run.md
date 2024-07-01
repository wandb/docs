# Run

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L461-L4184' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHubでソースを見る</a></button></p>

wandbによって記録される計算の単位。通常はML実験です。

```python
Run(
    settings: Settings,
    config: Optional[Dict[str, Any]] = None,
    sweep_config: Optional[Dict[str, Any]] = None,
    launch_config: Optional[Dict[str, Any]] = None
) -> None
```

`wandb.init()`を使ってrunを作成します：

```python
import wandb

run = wandb.init()
```

任意のプロセスにアクティブな`wandb.Run`は常に1つだけであり、それは`wandb.run`としてアクセス可能です：

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```

`wandb.log`で記録するすべてのデータはそのrunに送信されます。

同じスクリプトやノートブックでさらにRunを開始したい場合、実行中のrunを終了する必要があります。Runは`wandb.finish`を使って終了するか、`with`ブロック内で使用することで終了できます：

```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
    pass  # ここでデータを記録

assert wandb.run is None
```

Runの作成に関する詳細は`wandb.init`のドキュメントを参照するか、[`wandb.init`のガイド](https://docs.wandb.ai/guides/track/launch)をご覧ください。

分散トレーニングでは、ランク0プロセスで単一のrunを作成し、そのプロセスからのみ情報を記録するか、各プロセスでrunを作成し、それぞれから個別にログを記録し、`wandb.init`の`group`引数を使って結果をグループ化することができます。W&Bを使用した分散トレーニングの詳細については[ガイド](https://docs.wandb.ai/guides/track/log/distributed-training)をチェックしてください。

現在、`wandb.Api`には並列の`Run`オブジェクトがあります。最終的にはこれらのオブジェクトは統合される予定です。

| 属性 | 説明 |
| :--- | :--- |
| `summary` | (Summary) 各`wandb.log()`キーに対して設定された単一の値です。デフォルトでは、summaryは最後に記録された値に設定されます。最終値の代わりに、例えば最高の精度など、最良の値に手動で設定することもできます。 |
| `config` | このrunに関連するConfigオブジェクト。 |
| `dir` | runに関連するファイルが保存されるディレクトリ。 |
| `entity` | runに関連するW&Bエンティティの名前。エンティティはユーザー名やチームまたは組織の名前になることがあります。 |
| `group` | runに関連するグループの名前。グループを設定すると、W&B UIはrunを適切に整理します。分散トレーニングを行っている場合は、トレーニングのすべてのrunに同じグループを設定する必要があります。クロスバリデーションを行っている場合は、すべてのクロスバリデーションフォールドに同じグループを設定する必要があります。 |
| `id` | このrunの識別子。 |
| `mode` | `0.9.x`およびそれ以前との互換性のために、最終的には廃止予定。 |
| `name` | runの表示名。表示名は一意であることは保証されず、説明的なものにすることができます。デフォルトではランダムに生成されます。 |
| `notes` | runに関連するメモがある場合、そのメモ。メモは複数行の文字列で、マーキングや一部のLatex式を含むことができます。 |
| `path` | runへのパス。runのパスにはエンティティ、プロジェクト、run IDが含まれ、その形式は`entity/project/run_id`です。 |
| `project` | runに関連するW&Bプロジェクトの名前。 |
| `resumed` | runが再開された場合はTrue、それ以外はFalse。 |
| `settings` | runの設定のフローズンコピー。 |
| `start_time` | runの開始時刻のUnixタイムスタンプ（秒）。 |
| `starting_step` | runの最初のステップ。 |
| `step` | ステップの現在の値。このカウンタは`wandb.log`によってインクリメントされます。 |
| `sweep_id` | runに関連するSweepのID（存在する場合）。 |
| `tags` | runに関連するタグ（存在する場合）。 |
| `url` | runに関連するW&BのURL。 |

## メソッド

### `alert`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L3478-L3511)

```python
alert(
    title: str,
    text: str,
    level: Optional[Union[str, 'AlertLevel']] = None,
    wait_duration: Union[int, float, timedelta, None] = None
) -> None
```

与えられたタイトルとテキストでアラートを発動します。

| 引数 | 説明 |
| :--- | :--- |
| `title` | (str) アラートのタイトル。64文字以下でなければならない。 |
| `text` | (str) アラートの本文。 |
| `level` | (strまたはwandb.AlertLevel, オプション) 使用するアラートレベル。`INFO`, `WARN`, または`ERROR`のいずれか。 |
| `wait_duration` | (int, float, またはtimedelta, オプション) このタイトルで新たなアラートを送信するまでの待機時間（秒）。 |

### `define_metric`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2676-L2710)

```python
define_metric(
    name: str,
    step_metric: Union[str, wandb_metric.Metric, None] = None,
    step_sync: Optional[bool] = None,
    hidden: Optional[bool] = None,
    summary: Optional[str] = None,
    goal: Optional[str] = None,
    overwrite: Optional[bool] = None,
    **kwargs
) -> wandb_metric.Metric
```

`wandb.log()`で後に記録されるメトリクスの特性を定義します。

| 引数 | 説明 |
| :--- | :--- |
| `name` | メトリクスの名前。 |
| `step_metric` | メトリクスに関連する独立変数。 |
| `step_sync` | 必要に応じて`step_metric`を履歴に自動的に追加します。`step_metric`が指定された場合、デフォルトはTrue。 |
| `hidden` | このメトリクスを自動プロットから隠す。 |
| `summary` | summaryに追加する集計メトリクスを指定します。サポートされている集計方法："min,max,mean,best,last,none" デフォルトの集計は`copy`です。集計`best`のデフォルトは`goal`==`minimize`の場合です。 |
| `goal` | メトリクス最適化の方向を指定します。サポートされている方向："minimize,maximize" |

| 戻り値 | 説明 |
| :--- | :--- |
| メトリクスオブジェクトが返され、さらに指定できます。 |

### `detach`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2846-L2847)

```python
detach() -> None
```

### `display`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1349-L1357)

```python
display(
    height: int = 420,
    hidden: bool = (False)
) -> bool
```

このrunをjupyterで表示します。

### `finish`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2086-L2100)

```python
finish(
    exit_code: Optional[int] = None,
    quiet: Optional[bool] = None
) -> None
```

runを終了し、すべてのデータのアップロードを完了します。

これは同じプロセスで複数のrunを作成する場合に使用します。スクリプトが終了するときや、runコンテキストマネージャを使用するときに自動的にこのメソッドを呼び出します。

| 引数 | 説明 |
| :--- | :--- |
| `exit_code` | 0以外の値をセットしてrunを失敗としてマークします |
| `quiet` | ログ出力を最小限にするためにTrueを設定します |

### `finish_artifact`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L3096-L3148)

```python
finish_artifact(
    artifact_or_path: Union[Artifact, str],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    distributed_id: Optional[str] = None
) -> Artifact
```

非最終化アーティファクトをrunの出力として終了します。

同じdistributed_idを使用した後続の「upsert」は新しいバージョンを生成します。

| 引数 | 説明 |
| :--- | :--- |
| `artifact_or_path` | (strまたはArtifact) このアーティファクトの内容へのパス。以下の形式が使用できます： - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` または `wandb.Artifact`を呼び出して作成されたArtifactオブジェクトを渡すこともできます。 |
| `name` | (str, オプション) アーティファクトの名前。エンティティ/プロジェクトでプレフィックスされる場合がある。以下の形式が有効な名前です： - name:version - name:alias - digest これが指定されていない場合、デフォルトはパスのベース名に現在のrun IDを追加したものになります。 |
| `type` | (str) ログするアーティファクトのタイプ。`dataset`, `model`のような例を含む。 |
| `aliases` | (リスト, オプション) このアーティファクトに適用するエイリアス。デフォルトは `["latest"]`です。 |
| `distributed_id` | (string, オプション) すべての分散ジョブが共有する一意の文字列。Noneの場合、デフォルトはrunのグループ名です。 |

| 戻り値 | 説明 |
| :--- | :--- |
| `Artifact`オブジェクト。 |

### `get_project_url`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1231-L1239)

```python
get_project_url() -> Optional[str]
```

runに関連するW&BプロジェクトのURLを返します（存在する場合）。

オフラインのrunにはプロジェクトURLはありません。

### `get_sweep_url`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1241-L1246)

```python
get_sweep_url() -> Optional[str]
```

runに関連するSweepのURLを返します（存在する場合）。

### `get_url`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1221-L1229)

```python
get_url() -> Optional[str]
```

W&B runのURLを返します（存在する場合）。

オフラインのrunにはURLはありません。

### `join`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2134-L2144)

```python
join(
    exit_code: Optional[int] = None
) -> None
```

`finish()`の非推奨エイリアス - 代わりにfinishを使用してください。

### `link_artifact`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2849-L2895)

```python
link_artifact(
    artifact: Artifact,
    target_path: str,
    aliases: Optional[List[str]] = None
) -> None
```

与えられたアーティファクトをポートフォリオ（昇格されたアーティファクトのコレクション）にリンクします。

リンクされたアーティファクトは、指定されたポートフォリオのUIに表示されます。

| 引数 | 説明 |
| :--- | :--- |
| `artifact` | （公開またはローカルの）リンクされるアーティファクト |
| `target_path` | `{portfolio}、{project}/{portfolio}、または{entity}/{project}/{portfolio}`形式の文字列 |
| `aliases` | このポートフォリオ内のリンクされたアーティファクトにのみ適用されるエイリアスリスト。エイリアス「latest」は、リンクされたアーティファクトの最新バージョンに常に適用されます。 |

| 戻り値 | 説明 |
| :--- | :--- |
| None |

### `link_model`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L3384-L3476)

```python
link_model(
    path: StrPath,
    registered_model_name: str,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> None
```

モデルアーティファクトバージョンをログし、モデルレジストリ内の登録済みモデルにリンクします。

リンクされたモデルバージョンは、指定された登録済みモデルのUIに表示されます。

#### 手順:

- 'name'モデルアーティファクトがログされたかどうかを確認します。そうである場合には、'path'にあるファイルと一致するアーティファクトバージョンを使用するか、新しいバージョンをログします。そうでない場合は、'path'以下のファイルを新しいモデルアーティファクト 'name' としてログします。
- 'registered_model_name' という名前の登録済みモデルが 'model-registry' プロジェクトに存在するか確認します。存在しない場合、新しい登録済みモデル 'registered_model_name' を作成します。
- モデルアーティファクト 'name' のバージョンを登録済みモデル 'registered_model_name' にリンクします。
- 'aliases' リストから新しくリンクされたモデルアーティファクトバージョンにエイリアスを付加します。

| 引数 | 説明 |
| :--- | :--- |
| `path` | (str) モデルの内容へのパス。以下の形式が使用できます： - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
| `registered_model_name` | (str) - モデルがリンクされる登録済みモデルの名前。登録済みモデルは、通常チームの特定のMLタスクを表す、モデルバージョンのコレクションです。この登録済みモデルが属するエンティティはrun名から派生します: (str, optional) - 'path'のファイル内容がログされるモデルアーティファクトの名前。これが指定されない場合、デフォルトは現在のrun IDを前置したパスのベース名になります。 |
| `aliases` | (List[str], optional) - この登録済みモデル内でのみ適用されるエイリアス。エイリアス「latest」は、リンクされたアーティファクトの最新バージョンに常に適用されます。 |

#### 例:

```python
run.link_model(
    path="/local/directory",
    registered_model_name="my_reg_model",
    name="my_model_artifact",
    aliases=["production"],
)
```

無効な使用方法

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

| 引数 | 説明 |
| :--- | :--- |
| `AssertionError` | registered_model_nameがパスである場合、またはモデルアーティファクト'name'が'substring'を含まない場合に発生 |
| `ValueError` | nameに無効な特殊文字が含まれている場合に発生 |

| 戻り値 | 説明 |
| :--- | :--- |
| None |

### `log`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1665-L1877)

```python
log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None
) -> None
```

現在のrunの履歴にデータの辞書をログします。

`wandb.log`を使用して、ランからデータをログします。例えばスカラー、画像、ビデオ、ヒストグラム、プロット、テーブルなどがあります。

ライブ例、コードスニペット、ベストプラクティスなどの詳細については、[ログに関するガイド](https://docs.wandb.ai/guides/track/log)を参照してください。

最も基本的な使用法は`wandb.log({"train-loss": 0.5, "accuracy": 0.9})`です。
これにより、損失と精度がrunの履歴に保存され、これらのメトリクスのサマリ値が更新されます。

ログされたデータは[wandb.ai](https://wandb.ai)のワークスペースで可視化できます。ローカルでは[自己ホストインスタンス](https://docs.wandb.ai/guides/hosting)でも可視化できます。
Jupyterノートブックなどでローカルにデータをエクスポートして可視化および探索することもできます。[APIガイド](https://docs.wandb.ai/guides/track/public-api-guide)をご覧ください。

UIでは、サマリ値がrunテーブルに表示され、複数のrun間で単一の値を比較できます。
サマリ値は`wandb.run.summary["key"] = value`を使用して直接設定することもできます。

ログされた値は必ずしもスカラーである必要はありません。任意のwandbオブジェクトのログがサポートされています。
例えば、`wandb.log({"example": wandb.Image("myimage.jpg")})`は、W&B UIで適切に表示される例の画像をログします。
すべてのサポートされている型のリファレンスドキュメントを参照してください。
また、様々な例については[ログに関するガイド](https://docs.wandb.ai/guides/track/log)もチェックしてください。3D分子構造やセグメンテーションマスクからPRカーブやヒストグラムまで豊富な例があります。
`wandb.Table`は構造化データをログするために使用されます。詳細については[テーブルのログに関するガイド](https://docs.wandb.ai/guides/data-vis/log-tables)を参照してください。

ネストされたメトリクスのログは推奨され、W&B UIでサポートされています。
ネストされた辞書を使用して`wandb.log({"train": {"acc": 0.9}, "val": {"acc": 0.8}})`のようにログする場合、メトリクスはW&B UIの`train`と`val`のセクションに整理されます。

wandbはグローバルステップを追跡します。デフォルトでは`wandb.log`が呼び出されるごとにインクリメントされます。
関連するメトリクスを一緒にログすることが推奨されます。
関連するメトリクスを一緒にログするのが都合が悪い場合は、`wandb.log({"train-loss": 0.5}, commit=False)`を呼び出し、その後に`wandb.log({"accuracy": 0.9})`を呼び出すのは、`wandb.log({"train-loss": 0.5, "accuracy": 0.9})`を呼び出すのと同じです。

`wandb.log`は1秒間に数回以上呼び出すことは意図されていません。
それよりも頻繁にログしたい場合は、クライアント側でデータを集約するか、パフォーマンスが低下する可能性があります。

| 引数 | 説明 |
| :--- | :--- |
| `data` | (dict, オプション) シリアライズ可能なPythonオブジェクトの辞書。例えば、`str`, `ints`, `floats`, `Tensors`, `dicts` など、または `wandb.data_types` のいずれか。 |
| `commit` | (boolean, オプション) メトリクス辞書を wandb サーバーに保存し、ステップをインクリメントします。Falseの場合、`wandb.log`はdata引数を使用して現在のメトリクス辞書を更新し、`commit=True`で呼び出されるまでメトリクスは保存されません。 |
| `step` | (整数, オプション) プロセスのグローバルステップ。この非コミットされたステップの前にステップを保持しますが、指定されたステップをコミットしないのがデフォルトです。 |
| `sync` | (boolean, 真) この引数は非推奨であり、現在は`wandb.log`の動作を変更しません。 |

#### 例:

詳細な例については、[ログに関するガイド](https://docs.wandb.com/guides/track/log)を参照してください。

### 基本的な使い方

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
# 別のところでこのステップを報告する準備ができたら:
run.log({"accuracy": 0.8})
```

### ヒストグラム

```python
import numpy as np
import wandb

# 正規分布からランダムに勾配をサンプリング
gradients = np.random.randn(100, 100)
run = wandb.init()
run.log({"gradients": wandb.Histogram(gradients)})
```

### numpyからの画像

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

### PILからの画像

```python
import numpy as np
from PIL import Image as PILImage
import wandb

run = wandb.init()
examples = []
for i in range(3):
    pixels = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    pil_image = PILImage.fromarray(pixels, mode="RGB")
    image = wandb.Image(pil_image, caption=f"random field {i}")
    examples.append(image)
run.log({"examples": examples})
```

### numpyからのビデオ

```python
import numpy as np
import wandb

run = wandb.init()
# 軸は (time, channel, height, width)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
run.log({"video": wandb.Video(frames, fps=4)})
```

### Matplotlib Plot

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

run = wandb.init()
fig, ax = plt.subplots()
x = np.linspace(0, 10)
y = x * x
ax.plot(x, y)  # y = x^2 をプロット
run.log({"chart": fig})
```

### PR Curve

```python
import wandb

run = wandb.init()
run.log({"pr": wandb.plot.pr_curve(y_test, y_probas, labels)})
```

### 3D Object

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

| 引数 | 説明 |
| :--- | :--- |
| `wandb.Error` | `wandb.init`が呼び出される前に呼び出された場合 |
| `ValueError` | 無効なデータが渡された場合 |

### `log_artifact`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L3006-L3040)

```python
log_artifact(
    artifact_or_path: Union[Artifact, StrPath],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> Artifact
```

アーティファクトをrunの出力として宣言します。

| 引数 | 説明 |
| :--- | :--- |
| `artifact_or_path` | (strまたはArtifact) このアーティファクトの内容へのパス。以下の形式が使用できます： - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` または `wandb.Artifact`を呼び出して作成されたArtifactオブジェクトを渡すこともできます。 |
| `name` | (str, オプション) アーティファクト名。以下の形式が有効な名前です： - name:version - name:alias - digest これが指定されていない場合、デフォルトはパスのベース名に現在のrun IDを追加したものになります。 |
| `type` | (str) ログするアーティファクトのタイプ。`dataset`, `model`のような例を含む。 |
| `aliases` | (リスト, オプション) このアーティファクトに適用するエイリアス。デフォルトは `["latest"]`です。 |

| 戻り値 | 説明 |
| :--- | :--- |
| `Artifact`オブジェクト。 |

### `log_code`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1136-L1219)

```python
log_code(
    root: Optional[str] = ".",
    name: Optional[str] = None,
    include_fn: Union[Callable[[str, str], bool], Callable[[str], bool]] = _is_py_or_dockerfile,
    exclude_fn: Union[Callable[[str, str], bool], Callable[[str], bool]] = filenames.exclude_wandb_fn
) -> Optional[Artifact]
```

現在のコードの状態をW&B Artifactとして保存します。

デフォルトでは、現在のディレクトリを走査し、`.py`で終わるすべてのファイルをログします。

| 引数 | 説明 |
| :--- | :--- |
| `root` | リカーシブにコードを検索する相対（`os.getcwd()`に対する）または絶対パス。 |
| `name` | (str, オプション) コードアーティファクトの名前。デフォルトでは、アーティファクトの名前は `source-$PROJECT_ID-$ENTRYPOINT_RELPATH` になります。多くのrunが同じアーティファクトを共有したいシナリオでは、名前を指定することでこれを実現できます。 |
| `include_fn` | パスと（オプションで）ルートパスを受け取り、含めるべき場合にTrueを返し、それ以外の場合はFalseを返すコールバック。このデフォルトは： `lambda path, root: path.endswith(".py")` |
| `exclude_fn` | パスと（オプションで）ルートパスを受け取り、含めるべき場合にFalseを返し、それ以外の場合はTrueを返すコールバック。このデフォルトは`<root>/.wandb/` および `<root>/wandb/`ディレクトリ内のすべてのファイルを除外する関数です。 |

#### 例:

基本的な使い方

```python
run.log_code()
```

高度な使い方

```python
run.log_code(
    "../",
    include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
    exclude_fn=lambda path, root: os.path.relpath(path, root).startswith("cache/"),
)
```

| 戻り値 | 説明 |
| :--- | :--- |
| コードがログされた場合、`Artifact`オブジェクト |

### `log_model`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L3280-L3329)

```python
log_model(
    path: StrPath,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> None
```

runに含まれる'path'内の内容を含むモデルアーティファクトをログし、このrunの出力としてマークします。

| 引数 | 説明 |
| :--- | :--- |
| `path` | (str) モデルの内容へのパス。以下の形式が使用できます： - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
| `name` | (str, オプション) ファイル内容が追加されるモデルアーティファクトに割り当てる名前。この文字列は次の英数字文字のみを含む必要があります： ダッシュ（-）、アンダースコア（_）、およびドット（.）。指定されていない場合、デフォルトは現在のrun IDを前置したパスのベース名になります。 |
| `aliases` | (リスト, optional) 作成されたモデルアーティファクトに適用するエイリアス。デフォルトは`["latest"]`です。 |

#### 例:

```python
run.log_model(
    path="/local/directory",
    name="my_model_artifact",
    aliases=["production"],
)
```

無効な使用方法

```python
run.log_model(
    path="/local/directory",
    name="my_entity/my_project/my_model_artifact",
    aliases=["production"],
)
```

| 引数 | 説明 |
| :--- | :--- |
| `ValueError` | 名前に無効な特殊文字が含まれている場合に発生 |

| 戻り値 | 説明 |
| :--- | :--- |
| None |

### `mark_preempting`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L3529-L3537)

```python
mark_preempting() -> None
```

このrunをプリエンプティングとしてマークします。

また、すぐにサーバーにこれを報告するよう内部プロセスに伝えます。

### `plot_table`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2171-L2192)

```python
@staticmethod
plot_table(
    vega_spec_name: str,
    data_table: "wandb.Table",
    fields: Dict[str, Any],
    string_fields: Optional[Dict[str, Any]] = None,
    split_table: Optional[bool] = (False)
) -> CustomChart
```

テーブルに対してカスタムプロットを作成します。

| 引数 | 説明 |
| :--- | :--- |
| `vega_spec_name` | プロットのspecの名前 |
| `data_table` | 可視化に使用するデータを含むwandb.Tableオブジェクト |
| `fields` | テーブルキーからカスタム可視化に必要なフィールドへのマッピングを含む辞書 |
| `string_fields` | カスタム可視化に必要な文字列定数の値を提供する辞書 |

### `project_name`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1082-L1083)

```python
project_name() -> str
```

### `restore`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2071-L2084)

```python
restore(
    name: str,
    run_path: Optional[str] = None,
    replace: bool = (False),
    root: Optional[str] = None
) -> Union[None, TextIO]
```

指定されたファイルをクラウドストレージからダウンロードします。

ファイルは現在のディレクトリまたはrunディレクトリに配置されます。
デフォルトでは、ファイルが既に存在しない場合にのみダウンロードします。

| 引数 | 説明 |
| :--- | :--- |
| `name` | ファイルの名前 |
| `run_path` | ファイルをプルするためのrunへのパス。つまり、`username/project_name/run_id` もし`wandb.init`が呼び出されていない場合、これは必須です。 |
| `replace` | ファイルが既にローカルに存在する場合でもダウンロードするかどうか |
| `root` | ファイルをダウンロードするディレクトリ。デフォルトは現在のディレクトリまたは`wandb.init`が呼び出された場合のrunディレクトリ。 |

| 戻り値 | 説明 |
| :--- | :--- |
| ファイルが見つからない場合はNone、そうでなければ読み取り用にオープンされたファイルオブジェクト |

| 発生 | 説明 |
| :--- | :--- |
| `wandb.CommError` | wandbバックエンドに接続できない場合 |
| `ValueError` | ファイルが見つからない場合またはrun_pathを見つけられない場合 |

### `save`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1879-L1985)

```python
save(
    glob_str: Optional[Union[str, os.PathLike]] = None,
    base_path: Optional[Union[str, os.PathLike]] = None,
    policy: "PolicyName" = "live"
) -> Union[bool, List[str]]
```

1つまたは複数のファイルをW&Bに同期します。

相対パスは現在の作業ディレクトリを基準にします。

Unixグロブ（例えば「myfiles/*」）は`save`が呼び出された時に展開され、`policy` に関わりなく実行されます。特に、新しいファイルは自動的に検出されません。

`base_path` を提供してアップロードファイルのディレクトリ構造を制御できます。これは`glob_str`のプレフィックスであるべきで、以下の例を通して説明します：

```
wandb.save("these/are/myfiles/*")
# => ファイルをrunの「these/are/myfiles/」フォルダに保存。

wandb.save("these/are/myfiles/*", base_path="these")
# => ファイルをrunの「are/myfiles/」フォルダに保存。

wandb.save("/User/username/Documents/run123/*.txt")
# => ファイルをrunの「run123/」フォルダに保存。以下の注意点を参照。

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => ファイルをrunの「username/Documents/run123/」フォルダに保存。

wandb.save("files/*/saveme.txt")
# => 各「saveme.txt」ファイルを「files/」の適切なサブディレクトリに保存。
```

注意：絶対パスやグロブを指定しても`base_path`が無い場合、一つのディレクトリレベルが保持されることに注意してください。

| 引数 | 説明 |
| :--- | :--- |
| `glob_str` | 相対または絶対パスやUnixグロブ。 |
| `base_path` | ディレクトリ構造を推測するためのパス。 |
| `policy` | `live`, `now`, または `end`のいずれか * live: 変更が生じるたびにファイルをアップロードし、以前のバージョンを上書き * now: ファイルを今すぐ一度だけアップロード * end: runが終了したときにファイルをアップロード |

| 戻り値 | 説明 |
| :--- | :--- |
| マッチしたファイルのシンボリックリンクのパス。歴史的な理由から、従来のコードではブール値を返すこともあります。 |

### `status`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2146-L2169)

```python
status() -> RunStatus
```

現在のrunの同期状態について、内部バックエンドから情報を取得します。

### `to_html`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1359-L1368)

```python
to_html(
    height: int = 420,
    hidden: bool = (False)
) -> str
```

現在のrunを表示するiframeを含むHTMLを生成します。

### `unwatch`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2807-L2809)

```python
unwatch(
    models=None
) -> None
```

### `upsert_artifact`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L3042-L3094)

```python
upsert_artifact(
    artifact_or_path: Union[Artifact, str],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    distributed_id: Optional[str] = None
) -> Artifact
```

非最終化アーティファクトをrunの出力として宣言（または追加）します。

アーティファクトを最終化するには、run.finish_artifact()を呼び出す必要があります。これは分散ジョブが同じアーティファクトに貢献する必要がある場合に便利です。

| 引数 | 説明 |
| :--- | :--- |
| `artifact_or_path` | (strまたはArtifact) このアーティファクトの内容へのパス。以下の形式が使用できます： - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` または `wandb.Artifact`を呼び出して作成されたArtifactオブジェクトを渡すこともできます。 |
| `name` | (str, オプション) アーティファクトの名前。エンティティ/プロジェクトでプレフィックスされる場合がある。以下の形式が有効な名前です： - name:version - name:alias - digest これが指定されていない場合、デフォルトはパスのベース名に現在のrun IDを追加したものになります。 |
| `type` | (str) ログするアーティファクトのタイプ。`dataset`, `model`のような例を含む。 |
| `aliases` | (リスト, オプション) このアーティファクトに適用するエイリアス。デフォルトは `["latest"]`です。 |
| `distributed_id` | (string, オプション) すべての分散ジョブが共有する一意の文字列。Noneの場合、デフォルトはrunのグループ名です。 |

| 戻り値 | 説明 |
| :--- | :--- |
| `Artifact`オブジェクト。 |

### `use_artifact`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2897-L3004)

```python
use_artifact(
    artifact_or_name: Union[str, Artifact],
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    use_as: Optional[str] = None
) -> Artifact
```

アーティファクトをrunの入力として宣言します。

戻りオブジェクトで `download` または `file` を呼び出して、コンテンツをローカルに取得します。

| 引数 | 説明 |
| :--- | :--- |
| `artifact_or_name` | (strまたはArtifact) アーティファクト名。エンティティ/プロジェクトでプレフィックスされる場合がある。以下の形式が有効な名前です： - name:version - name:alias または `wandb.Artifact`を呼び出して作成されたアーティファクトオブジェクトを渡すこともできます |
| `type` | (str, オプション) 使用するアーティファクトのタイプ。 |
| `aliases` | (リスト, オプション) このアーティファクトに適用するエイリアス |
| `use_as` | (string, オプション) アーティファクトの使用目的を示す文字列。UIで表示されます。 |

| 戻り値 | 説明 |
| :--- | :--- |
| `Artifact`オブジェクト。 |

### `use_model`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L3331-L3382)

```python
use_model(
    name: str
) -> FilePathStr
```

モデルアーティファクト 'name' にログされたファイルをダウンロードします。

| 引数 | 説明 |
| :--- | :--- |
| `name` | (str) モデルアーティファクト名。'name'は既存のログ済みモデルアーティファクトの名前と一致する必要があります。エンティティ/プロジェクトでプレフィックスされる場合がある。以下の形式が有効な名前です： - model_artifact_name:version - model_artifact_name:alias |

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

無効な使用方法

```python
run.use_model(
    name="my_entity/my_project/my_model_artifact",
)
```

| 引数 | 説明 |
| :--- | :--- |
| `AssertionError` | モデルアーティファクト 'name' が 'model' というサブストリングを含まない型である場合に発生。 |

| 戻り値 | 説明 |
| :--- | :--- |
| `path` | (str) ダウンロードされたモデルアーティファクトファイルへのパス。 |

### `watch`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2794-L2804)

```python
watch(
    models, criterion=None, log="gradients", log_freq=100, idx=None,
    log_graph=(False)
) -> None
```

### `__enter__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L3513-L3514)

```python
__enter__() -> "Run"
```

### `__exit__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L3516-L3527)

