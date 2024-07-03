# Run

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L461-L4184' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

wandb によってログされた計算の単位。通常、これは ML 実験です。

```python
Run(
    settings: Settings,
    config: Optional[Dict[str, Any]] = None,
    sweep_config: Optional[Dict[str, Any]] = None,
    launch_config: Optional[Dict[str, Any]] = None
) -> None
```

`wandb.init()` で run を作成します:

```python
import wandb

run = wandb.init()
```

どのプロセスにおいてもアクティブな `wandb.Run` は最大で1つだけであり、それは `wandb.run` としてアクセスできます:

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```

`wandb.log` でログするすべてのものがその run に送信されます。

同じスクリプトやノートブックで複数の run を開始したい場合、現在進行中の run を終了する必要があります。Runs は `wandb.finish` で終了させるか、`with` ブロック内で使用して終了させることができます:

```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
    pass  # ここでデータをログ
```

run の作成に関する詳細は `wandb.init` のドキュメントを参照するか、[wandb.init のガイド](https://docs.wandb.ai/guides/track/launch) をチェックしてください。

分散トレーニングでは、rank 0 プロセスで単一の run を作成し、そのプロセスからのみ情報をログするか、各プロセスで run を作成し、それぞれから別々にログし、結果を `wandb.init` の `group` 引数を使用して一緒にグループ化することができます。W&B を使用した分散トレーニングに関する詳細は、[ガイド](https://docs.wandb.ai/guides/track/log/distributed-training) を参照してください。

現在、`wandb.Api` に並行して `Run` オブジェクトがあります。最終的にはこれら二つのオブジェクトは統合される予定です。

| 属性 |  |
| :--- | :--- |
|  `summary` |  (Summary) 各 `wandb.log()` キーに設定された単一の値。デフォルトでは、summary は最後にログされた値に設定されます。最終値ではなく最大精度などのベスト値に手動で設定することもできます。|
|  `config` |  この run に関連付けられた config オブジェクト。 |
|  `dir` |  run に関連付けられたファイルが保存されるディレクトリ。 |
|  `entity` |  run に関連付けられた W&B の entity の名前。entity はユーザー名またはチームや組織の名前です。 |
|  `group` |  run に関連付けられたグループの名前。group を設定すると、W&B UI が runs をわかりやすく整理します。分散トレーニングを行っている場合、トレーニング内のすべての run に同じ group を指定する必要があります。交差検証を行っている場合、すべての交差検証フォールドに同じ group を指定する必要があります。 |
|  `id` |  この run の識別子。 |
|  `mode` |  `0.9.x` およびそれ以前との互換性のため、最終的には非推奨。 |
|  `name` |  run の表示名。表示名は一意であることが保証されておらず、記述的である可能性があります。デフォルトではランダムに生成されます。 |
|  `notes` |  run に関連付けられたノートがある場合、それらのノート。ノートは複数行の文字列で、markdown と latex 式を `$$` 内に使用できます。 |
|  `path` |  run へのパス。run パスは以下の形式で entity、project、および run ID を含みます: `entity/project/run_id`。 |
|  `project` |  run に関連付けられた W&B プロジェクトの名前。 |
|  `resumed` |  run が再開された場合は True、それ以外の場合は False。 |
|  `settings` |  run の Settings オブジェクトの凍結コピー。 |
|  `start_time` |  run が開始された時点の Unix タイムスタンプ（秒）。 |
|  `starting_step` |  run の最初のステップ。 |
|  `step` |  現在のステップの値。このカウンターは `wandb.log` によってインクリメントされます。 |
|  `sweep_id` |  run に関連付けられた sweep の ID（もしあれば）。 |
|  `tags` |  run に関連付けられたタグ（もしあれば）。 |
|  `url` |  run に関連付けられた W&B の URL。 |

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

指定されたタイトルとテキストでアラートを起動します。

| 引数 |  |
| :--- | :--- |
|  `title` |  (str) アラートのタイトル、64文字以内でなければなりません。 |
|  `text` |  (str) アラートの本文。 |
|  `level` |  (str または wandb.AlertLevel、オプション) 使用するアラートレベル、`INFO`、`WARN`、または `ERROR` のいずれか。 |
|  `wait_duration` |  (int, float、または timedelta、オプション) このタイトルのアラートを再送信する前に待つ時間（秒）。 |

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

後に `wandb.log()` でログされるメトリクスのプロパティを定義します。

| 引数 |  |
| :--- | :--- |
|  `name` |  メトリクスの名前。 |
|  `step_metric` |  メトリクスに関連付けられた独立変数。 |
|  `step_sync` |  必要に応じて `step_metric` を履歴に自動的に追加します。step_metric が指定された場合、デフォルトは True です。 |
|  `hidden` |  このメトリクスを自動プロットから非表示にします。 |
|  `summary` |  summary に追加された集計メトリクスを指定します。サポートされている集計: "min, max, mean, best, last, none" デフォルトの集計は `copy` です。集計の `best` はデフォルトで `goal`==`minimize` です。 |
|  `goal` |  メトリクスの最適化方向を指定します。サポートされている方向: "minimize, maximize" |

| 戻り値 |  |
| :--- | :--- |
|  さらなる指定が可能なメトリクスオブジェクトが返されます。 |

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

この run を jupyter に表示します。

### `finish`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2086-L2100)

```python
finish(
    exit_code: Optional[int] = None,
    quiet: Optional[bool] = None
) -> None
```

run を終了し、すべてのデータのアップロードを終了します。

これは同じプロセスで複数の run を作成する場合に使用します。スクリプトが終了するか run コンテキストマネージャを使用すると、このメソッドが自動的に呼び出されます。

| 引数 |  |
| :--- | :--- |
|  `exit_code` |  0 以外の値を設定して run を失敗と見なします |
|  `quiet` |  ログの出力を最小限にする場合は true に設定します |

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

非最終化アーティファクトを run の出力として終了します。

同じ分散 ID の後続の "アップサート" は新しいバージョンを生成します。

| 引数 |  |
| :--- | :--- |
|  `artifact_or_path` |  (str または Artifact) このアーティファクトの内容へのパス。以下の形式が使えます: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` または `wandb.Artifact` の作成で得られた Artifact オブジェクトを渡すこともできます。 |
|  `name` |  (str, オプション) アーティファクト名。entity/project の接頭辞を持つ名前です。以下の形式が有効です: - name:version - name:alias - digest 指定されない場合は、パスのベース名に現在の run id を追加したものがデフォルトになります。 |
|  `type` |  (str) ログするアーティファクトのタイプ。例として `dataset`, `model` があります。 |
|  `aliases` |  (リスト, オプション) このアーティファクトに適用するエイリアス。デフォルトは `["latest"]` です。 |
|  `distributed_id` |  (string, オプション) 分散ジョブが共有する一意の文字列。None の場合、デフォルトは run の group 名になります。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `get_project_url`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1231-L1239)

```python
get_project_url() -> Optional[str]
```

run に関連付けられた W&B プロジェクトの URL を返します（もし存在すれば）。

オフライン run にはプロジェクト URL がありません。

### `get_sweep_url`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1241-L1246)

```python
get_sweep_url() -> Optional[str]
```

run に関連する sweep の URL を返します（もし存在すれば）。

### `get_url`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1221-L1229)

```python
get_url() -> Optional[str]
```

W&B run の URL を返します（もし存在すれば）。

オフライン run には URL がありません。

### `join`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2134-L2144)

```python
join(
    exit_code: Optional[int] = None
) -> None
```

`finish()` の非推奨エイリアスです - 代わりに finish を使用してください。

### `link_artifact`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2849-L2895)

```python
link_artifact(
    artifact: Artifact,
    target_path: str,
    aliases: Optional[List[str]] = None
) -> None
```

指定されたアーティファクトをポートフォリオ（アーティファクトの昇格されたコレクション）にリンクします。

リンクされたアーティファクトは、指定したポートフォリオの UI で表示されます。

| 引数 |  |
| :--- | :--- |
|  `artifact` |  リンクされる（公開またはローカルの）アーティファクト。 |
|  `target_path` |  `str` - 以下の形式のいずれかを取る: {portfolio}, {project}/{portfolio}, または {entity}/{project}/{portfolio} |
|  `aliases` |  `List[str]` - ポートフォリオ内でこのリンクされたアーティファクトにのみ適用されるオプションのエイリアス。エイリアス "latest" は、リンクされたアーティファクトの最新バージョンに常に適用されます。 |

| 戻り値 |  |
| :--- | :--- |
|  なし |

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

モデルアーティファクトバージョンをログしてモデルレジストリ内の登録済みモデルにリンクします。

リンクされたモデルバージョンは、指定された登録済みモデルの UI に表示されます。

#### 手順:

- 'name' モデルアーティファクトがログされているかどうかを確認します。そうであれば、'path' にあるファイルが一致するアーティファクトバージョンを使用するか、新しいバージョンをログします。そうでない場合、'path' にあるファイルを使って新しいモデルアーティファクト 'name' をログします。
- 'registered_model_name' 名を持つ登録済みモデルが 'model-registry' プロジェクト内に存在するか確認します。存在しない場合、新しい登録済みモデル 'registered_model_name' を作成します。
- 'name' モデルアーティファクトのバージョンを登録済みモデル 'registered_model_name' にリンクします。
- 'aliases' リストから新しくリンクされたモデルアーティファクトバージョンにエイリアスを適用します。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) このモデルの内容へのパス。以下の形式が使えます: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `registered_model_name` |  (str) - モデルレジストリにリンクする登録済みモデルの名前。登録済みモデルはモデルレジストリにリンクされたモデルバージョンのコレクションであり、通常、チームの特定のMLタスクを表します。登録済みモデルが付属するエンティティはrun名から導き出されます : (str, オプション) - 'path'内のファイルがログされるモデルアーティファクト名。指定されていない場合、デフォルトでパスのベース名に現在のrun idが付加されます。 |
|  `aliases` |  (リスト, オプション) - この登録済みモデル内のリンクされたアーティファクトにのみ適用されるエイリアス。エイリアス "latest" は、リンクされたアーティファクトの最新バージョンに常に適用されます。 |

#### 例:

```python
run.link_model(
    path="/local/directory",
    registered_model_name="my_reg_model",
    name="my_model_artifact",
    aliases=["production"],
)
```

無効な使用例

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

| 例外 |  |
| :--- | :--- |
|  `AssertionError` |  `registered_model_name` がパスである場合、または 'name' のモデルアーティファクトが 'model' という文字列を含まないタイプである場合 |
|  `ValueError` |  名前に無効な特殊文字が含まれている場合 |

| 戻り値 |  |
| :--- | :--- |
|  なし |

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

現在の run の履歴にデータの辞書をログします。

`wandb.log` を使用して run からデータをログし、スカラー、画像、ビデオ、ヒストグラム、プロット、テーブルなどを記録します。

ライブサンプル、コードスニペット、ベストプラクティスなどについては、[ログインガイド](https://docs.wandb.ai/guides/track/log) を参照してください。

最も基本的な使用法は `wandb.log({"train-loss": 0.5, "accuracy": 0.9})` です。
これは損失と精度を run の履歴に保存し、これらのメトリクスのサマリ値を更新します。

[wandb.ai](https://wandb.ai) のワークスペースでログされたデータを視覚化するか、
W&B アプリの[セルフホストインスタンス](https://docs.wandb.ai/guides/hosting) でローカルにし、
あるいはデータをエクスポートしてローカルで視覚化し、Jupyter ノートブックなどで探ります。
詳細は、[API ガイド](https://docs.wandb.ai/guides/track/public-api-guide) を参照してください。

UI では、サマリ値が run テーブルに表示され、run間で単一の値を比較できます。
サマリ値は `wandb.run.summary["key"] = value` を使用して直接設定することもできます。

ログ値はスカラーである必要はありません。任意の wandb オブジェクトのログがサポートされています。
例えば、`wandb.log({"example": wandb.Image("myimage.jpg")})` とすると、例の画像がログされ、W&B UI にうまく表示されます。
サポートされているすべてのタイプについては、[リファレンスドキュメント](https://docs.wandb.com/ref/python/data-types) を参照するか、
3D 分子構造やセグメンテーションマスク、PRカーブ、ヒストグラムに至るまでの例については、[ログガイド](https://docs.wandb.ai/guides/track/log) を参照してください。
`wandb.Table` を使用して構造化されたデータをログすることもできます。
詳細は [テーブルのログガイド](https://docs.wandb.ai/guides/data-vis/log-tables) を参照してください。

ネストされたメトリクスのログは推奨されており、W&B UI でもサポートされています。
`wandb.log({"train": {"acc": 0.9}, "val": {"acc": 0.8}})` のように
ネストされた辞書を使用してログすると、メトリクスは W&B UI の `train` と `val` セクションに整理されます。

wandb はグローバルステップを追跡し、デフォルトでは `wandb.log`
を呼び出すたびに増加しますので、関連するメトリクスを一緒にログすることをお勧めします。
関連するメトリクスを一緒にログするのが不便な場合、次の2つの呼び出しは同等です:

`wandb.log({"train-loss": 0.5}, commit=False)` と次に

`wandb.log({"accuracy": 0.9})` と呼び出すことは

`wandb.log({"train-loss": 0.5, "accuracy": 0.9})` と呼び出すこと。

`wandb.log` は1秒につき数回以上呼び出されることを意図していません。
それ以上の頻度でログを取りたい場合は、クライアント側でデータを集約する方が良く、そうしないとパフォーマンスが低下する可能性があります。

| 引数 |  |
| :--- | :--- |
|  `data` |  (dict, オプション) シリアライズ可能な Python オブジェクトの辞書 i.e `str`, `ints`, `floats`, `Tensors`, `dicts`, または `wandb.data_types` のいずれか。 |
|  `commit` |  (boolean, オプション) メトリクス辞書を wandb サーバーに保存し、ステップを増加させます。false に設定すると、`wandb.log` は現在のメトリクス辞書をデータ引数で更新し、`commit=True` が呼び出されるまでメトリクスは保存されません。 |
|  `step` |  (整数, オプション) プロセッシング内のグローバルステップ。この引数が指定されても、デフォルトでは指定されたステップをコミットしません。 |
|  `sync` |  (boolean, True) この引数は非推奨であり、現在 `wandb.log` の動作を変更しません。 |

#### 例:

詳細でより多くの例については、[ログインガイド](https://docs.wandb.com/guides/track/log) を参照してください。

### 基本的な使い方:

```python
import wandb

run = wandb.init()
run.log({"accuracy": 0.9, "epoch": 5})
```

### インクリメンタルログ:

```python
import wandb

run = wandb.init()
run.log({"loss": 0.2}, commit=False)
# レポートする準備ができたとき:
run.log({"accuracy": 0.8})
```

### ヒストグラム:

```python
import numpy as np
import wandb

# 正規分布からランダムにサンプルした勾配
gradients = np.random.randn(100, 100)
run = wandb.init()
run.log({"gradients": wandb.Histogram(gradients)})
```

### NumPy からの画像:

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

### PIL からの画像:

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

### NumPy からのビデオ:

```python
import numpy as np
import wandb

run = wandb.init()
# 軸は (time, channel, height, width)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
run.log({"video": wandb.Video(frames, fps=4)})
```

### Matplotlib プロット:

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

### PR 曲線:

```python
import wandb

run = wandb.init()
run.log({"pr": wandb.plot.pr_curve(y_test, y_probas, labels)})
```

### 3D オブジェクト:

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

| 例外 |  |
| :--- | :--- |
|  `wandb.Error` |  `wandb.init` を呼び出す前に呼び出された場合 |
|  `ValueError` |  無効なデータが渡された場合 |

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

アーティファクトを run の出力として宣言します。

| 引数 |  |
| :--- | :--- |
|  `artifact_or_path` |  (str または Artifact) アーティファクトの内容へのパス。以下の形式が使えます: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` または `wandb.Artifact` の作成で得られた Artifact オブジェクトを渡すこともできます。 |
|  `name` |  (str, オプション) アーティファクト名。以下の形式が有効です: - name:version - name:alias - digest 指定されない場合は、パスのベース名に現在の run id を追加したものがデフォルトになります。 |
|  `type` |  (str) ログするアーティファクトのタイプ。例として `dataset`, `model` があります。 |
|  `aliases` |  (リスト, オプション) このアーティファクトに適用するエイリアス。デフォルトは `["latest"]` です。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

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

現在のコードの状態を W&B アーティファクトに保存します。

デフォルトでは、現在のディレクトリを巡回し、`.py` で終わるすべてのファイルをログします。

| 引数 |  |
| :--- | :--- |
|  `root` |  コードを再帰的に見つけるための `os.getcwd()` に対する相対パスまたは絶対パス。 |
|  `name` |  (str, オプション) コードアーティファクトの名前。デフォルトでは、アーティファクトの名前を `source-$PROJECT_ID-$ENTRYPOINT_RELPATH` とします。多くの run が同じアーティファクトを共有するシナリオがあるかもしれません。name を指定することでそれを達成できます。 |
|  `include_fn` |  ファイルパスと（オプションで）ルートパスを受け取り、それを含めるべきときに True を返し、それ以外の場合は False を返す callable。このデフォルトは: `lambda path, root: path.endswith(".py")` |
|  `exclude_fn` |  ファイルパスと（オプションで）ルートパスを受け取り、除外すべきときは `True` を返し、それ以外の場合は False を返す callable。このデフォルトは `&lt;root&gt;/.wandb/` および `&lt;root&gt;/wandb/` ディレクトリ内のすべてのファイルを除外します。 |

#### 例:

基本的な使い方

```python
run.log_code()
```

高度な使用法

```python
run.log_code(
    "../",
    include_fn=lambda path: path.endswith(".py") または path.endswith(".ipynb"),
    exclude_fn=lambda path, root: os.path.relpath(path, root).startswith("cache/"),
)
```

| 戻り値 |  |
| :--- | :--- |
|  コードがログされた場合は `Artifact` オブジェクト |

### `log_model`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L3280-L3329)

```python
log_model(
    path: StrPath,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> None
```

'path' 内の内容を含むモデルアーティファクトを run にログし、それをこの run の出力としてマークします。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) このモデルの内容へのパス。以下の形式が使えます: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `name` |  (str, オプション) ファイル内容が追加されるモデルアーティファクトに割り当てる名前。文字列には以下の英数字、ダッシュ、アンダースコア、およびドットのみを含めることができます。指定されていない場合、デフォルトでパスのベース名に現在の run id を追加したものになります。 |
|  `aliases` |  (リスト, オプション) 作成されたモデルアーティファクトに適用するエイリアス。デフォルトは `["latest"]` です。 |

#### 例:

```python
run.log_model(
    path="/local/directory",
    name="my_model_artifact",
    aliases=["production"],
)
```

無効な使用例

```python
run.log_model(
    path="/local/directory",
    name="my_entity/my_project/my_model_artifact",
    aliases=["production"],
)
```

| 例外 |  |
| :--- | :--- |
|  `ValueError` |  名前に無効な特殊文字が含まれている場合 |

| 戻り値 |  |
| :--- | :--- |
|  なし |

### `mark_preempting`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L3529-L3537)

```python
mark_preempting() -> None
```

この run をプリエンプティングとしてマークします。

また、内部プロセスにこれをサーバーに即座に報告するよう指示します。

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

テーブルにカスタムプロットを作成します。

| 引数 |  |
| :--- | :--- |
|  `vega_spec_name` |  プロットの spec 名 |
|  `data_table` |  可視化に使用されるデータを含む wandb.Table オブジェクト |
|  `fields` |  カスタム可視化が必要とするフィールドにテーブルキーをマッピングする辞書 |
|  `string_fields` |  カスタム可視化が必要とする任意の文字列定数の値を提供する辞書 |

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

クラウドストレージから指定されたファイルをダウンロードします。

ファイルは現在のディレクトリまたは実行ディレクトリに配置されます。
デフォルトでは、ファイルがすでに存在しない場合にのみダウンロードされます。

| 引数 |  |
| :--- | :--- |
|  `name` |  ファイルの名前 |
|  `run_path` |  ファイルを取得する実行のパス、例: `username/project_name/run_id`。wandb.init が呼び出されていない場合、これは必須です。 |
|  `replace` |  ローカルにすでに存在する場合でもファイルをダウンロードするかどうか |
|  `root` |  ファイルをダウンロードするディレクトリ。デフォルトは現在のディレクトリまたは wandb.init が呼び出された場合は実行ディレクトリです。 |

| 戻り値 |  |
| :--- | :--- |
|  ファイルが見つからない場合は None、それ以外の場合は読み取り用に開かれたファイルオブジェクト |

| 例外 |  |
| :--- | :--- |
|  `wandb.CommError` |  wandb バックエンドに接続できない場合 |
|  `ValueError` |  ファイルが見つからない場合や run_path が見つからない場合 |

### `save`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1879-L1985)

```python
save(
    glob_str: Optional[Union[str, os.PathLike]] = None,
    base_path: Optional[Union[str, os.PathLike]] = None,
    policy: "PolicyName" = "live"
) -> Union[bool, List[str]]
```

1つまたは複数のファイルを W&B と同期します。

相対パスは現在のワーキングディレクトリに相対的です。

Unix グロブ（例えば "myfiles/*"）は `save` が呼び出された時点で展開されます。`policy` に関わらず、特に新しいファイルは自動的には取得されません。

`base_path` を指定することで、アップロードされたファイルのディレクトリ構造を制御できます。それは `glob_str` のプレフィックスであるべきであり、その下のディレクトリ構造は保存されます。以下の例で理解してください:

```
wandb.save("these/are/myfiles/*")
# => run 内で "these/are/myfiles/" フォルダにファイルが保存されます。

wandb.save("these/are/myfiles/*", base_path="these")
# => run 内で "are/myfiles/" フォルダにファイルが保存されます。

wandb.save("/User/username/Documents/run123/*.txt")
# => run 内で "run123/" フォルダにファイルが保存されます。以下の注意を参照してください。

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => run 内で "username/Documents/run123/" フォルダにファイルが保存されます。

wandb.save("files/*/saveme.txt")
# => 各 "saveme.txt" ファイルがそれぞれのサブディレクトリ "files/" に保存されます。
```

注意: 絶対パスやグロブが指定され、`base_path` がない場合、1つのディレクトリレベルが保存されます。

| 引数 |  |
| :--- | :--- |
|  `glob_str` |  相対パスまたは絶対パスまたは Unix グロブ。 |
|  `base_path` |  ディレクトリ構造を推測するためのパス。詳細は例を参照。 |
|  `policy` |  `live`, `now`, または `end` のいずれか: * live: ファイルが変更されるたびにアップロードし、以前のバージョンを上書きする * now: 今すぐファイルをアップロード * end: run が終了したときにファイルをアップロード |

| 戻り値 |  |
| :--- | :--- |
|  マッチしたファイルのシンボリックリンクパス。一部の古いコードではブール値を返す場合があります。 |

### `status`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L2146-L2169)

```python
status() -> RunStatus
```

現在の run の同期ステータスについて、内部バックエンドからの情報を取得します。

### `to_html`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1359-L1368)

```python
to_html(
    height: int = 420,
    hidden: bool = (False)
) -> str
```

現在の run を表示する iframe を含む HTML を生成します。

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

非最終化アーティファクトを run の出力として宣言または追加します。

run.finish_artifact() を呼び出してアーティファクトを最終化する必要があります。
これは分散ジョブが同じアーティファクトに貢献する場合に役立ちます。

| 引数 |  |
| :--- | :--- |
|  `artifact_or_path` |  (str または Artifact) アーティファクトの内容へのパス。以下の形式が使えます: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` または `wandb.Artifact` の作成で得られた Artifact オブジェクトを渡すこともできます。 |
|  `name` |  (str, オプション) アーティファクト名。entity/project の接頭辞を持つ名前です。以下の形式が有効です: - name:version - name:alias - digest 指定されない場合は、パスのベース名に現在の run id を追加したものがデフォルトになります。 |
|  `type` |  (str) ログするアーティファクトのタイプ。例として `dataset`, `model` があります。 |
|  `aliases` |  (リスト, オプション) このアーティファクトに適用するエイリアス。デフォルトは `["latest"]` です。 |
|  `distributed_id` |  (string, オプション) 分散ジョブが共有する一意の文字列。None の場合、デフォルトは run の group 名になります。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

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

アーティファクトを run の入力として宣言します。

戻り値のオブジェクトで `download` または `file` を呼び出して、内容をローカルに取得します。

| 引数 |  |
| :--- | :--- |
|  `artifact_or_name` |  (str または Artifact) アーティファクト名。entity/project のプレフィックスを持つ名前です。以下の形式が有効です: - name:version - name:alias `wandb.Artifact` を呼び出すことで作成された Artifact オブジェクトを渡すこともできます。 |
|  `type` |  (str, オプション) 使用するアーティファクトのタイプ。 |
|  `aliases` |  (リスト, オプション) このアーティファクトに適用するエイリアス。 |
|  `use_as` |  (string, オプション) アーティファクトの使用目的を示すオプションの文字列。UI に表示されます。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `use_model`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L3331-L3382)

```python
use_model(
    name: str
) -> FilePathStr
```

モデルアーティファクト 'name' にログされたファイルをダウンロードします。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) モデルアーティファクト名。'name' は既存のログされたモデルアーティファクトの名前と一致する必要があります。entity/project のプレフィックスを持つ名前です。以下の形式が有効です: - model_artifact_name:version - model_artifact_name:alias |

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

無効な使用例

```python
run.use_model(
    name="my_entity/my_project/my_model_artifact",
)
```

| 例外 |  |
| :--- | :--- |
|  `AssertionError` |  モデルアーティファクト 'name' が 'model' という文字列を含まないタイプの場合 |

| 戻り値 |  |
| :--- | :--- |
|  `path` |  (str) ダウンロードされたモデルアーティファクトファイルへのパス。 |

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

```python
__exit__(
    exc_type: Type[BaseException],
    exc_val: BaseException,
    exc_tb: TracebackType
) -> bool
```