
# Run

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L461-L4183' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

wandb によってログされる計算の単位です。通常、これは ML 実験です。

```python
Run(
    settings: Settings,
    config: Optional[Dict[str, Any]] = None,
    sweep_config: Optional[Dict[str, Any]] = None,
    launch_config: Optional[Dict[str, Any]] = None
) -> None
```

`wandb.init()` を使用して run を作成します:

```python
import wandb

run = wandb.init()
```

任意のプロセスでアクティブな `wandb.Run` は最大で一つだけであり、それは `wandb.run` としてアクセスできます:

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```

`wandb.log` でログするすべてがその run に送信されます。

同じスクリプトやノートブックで複数の run を開始したい場合、進行中の run を完了する必要があります。run は `wandb.finish` を使用するか、`with` ブロックで使用することで完了できます:

```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
    pass  # ここでデータをログします

assert wandb.run is None
```

run の作成に関する詳細は `wandb.init` のドキュメントをご覧ください。または [wandb.init に関するガイド](https://docs.wandb.ai/guides/track/launch) を確認してください。

分散トレーニングでは、ランク0のプロセスで単一の run を作成し、そのプロセスからだけ情報をログするか、または各プロセスで run を作成し、それぞれからログを取り、その結果を `wandb.init` の `group` 引数でグループ化することができます。W&B を使用した分散トレーニングの詳細については、[ガイド](https://docs.wandb.ai/guides/track/log/distributed-training) をご覧ください。

現在、`wandb.Api` に並行する `Run` オブジェクトがあります。最終的にはこれらのオブジェクトは統合される予定です。

| 属性 |  |
| :--- | :--- |
|  `summary` |  (Summary) 各 `wandb.log()` キーに設定された単一の値。デフォルトでは、summary は最後にログされた値に設定されます。最大の精度など、最良の値を手動で設定することもできます。 |
|  `config` |  この run に関連付けられた Config オブジェクト。 |
|  `dir` |  run に関連するファイルが保存されるディレクトリー。 |
|  `entity` |  run に関連付けられた W&B エンティティの名前。エンティティはユーザー名、チーム名、または組織の名前である場合があります。 |
|  `group` |  run に関連付けられたグループの名前。グループを設定すると、W&B UI が run を整理しやすくなります。分散トレーニングを行う場合は、トレーニングのすべての run に同じグループ名を指定してください。クロスバリデーションを行う場合は、すべてのクロスバリデーションフォールドに同じグループ名を指定してください。 |
|  `id` |  この run の識別子。 |
|  `mode` |  `0.9.x` およびそれ以前との互換性のためにありますが、最終的には廃止されます。 |
|  `name` |  run の表示名。表示名は一意であることが保証されず、説明的である場合があります。デフォルトでは、ランダムに生成されます。 |
|  `notes` |  run に関連付けられたノートがあればそれを表示します。ノートは複数行の文字列であり、markdown や latex の数式も含むことができます。 |
|  `path` |  run へのパス。Run パスにはエンティティ、プロジェクト、および run ID が含まれ、形式は `entity/project/run_id` です。 |
|  `project` |  run に関連付けられている W&B プロジェクトの名前。 |
|  `resumed` |  run が再開された場合は True、それ以外の場合は False。 |
|  `settings` |  run の Settings オブジェクトの凍結版。 |
|  `start_time` |  run が開始された時点の Unix タイムスタンプ（秒単位）。 |
|  `starting_step` |  run の最初のステップ。 |
|  `step` |  現在のステップの値。このカウンターは `wandb.log` によってインクリメントされます。 |
|  `sweep_id` |  run に関連付けられている sweep の ID があればそれを表示します。 |
|  `tags` |  run に関連付けられているタグがあればそれを表示します。 |
|  `url` |  run に関連付けられている W&B の URL。 |

## メソッド

### `alert`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L3480-L3513)

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
|  `title` |  (str) アラートのタイトル、64文字以内である必要があります。 |
|  `text` |  (str) アラートの本文。 |
|  `level` |  (str または wandb.AlertLevel、オプション) 使用するアラートレベル、`INFO`, `WARN`, または `ERROR` のいずれか。 |
|  `wait_duration` |  (int, float, または timedelta、オプション) このタイトルで別のアラートを送信する前に待機する時間（秒単位）。 |

### `define_metric`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2681-L2715)

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

後で `wandb.log()` でログされるメトリクスのプロパティを定義します。

| 引数 |  |
| :--- | :--- |
|  `name` |  メトリクスの名前。 |
|  `step_metric` |  メトリクスに関連付けられた独立変数。 |
|  `step_sync` |  必要に応じて `step_metric` を自動的に履歴に追加します。`step_metric` が指定されていればデフォルトで True になります。 |
|  `hidden` |  このメトリクスを自動プロットから非表示にします。 |
|  `summary` |  summary に追加される集計メトリクスを指定します。サポートされている集計: "min,max,mean,best,last,none" デフォルトの集計は `copy` です。集計 `best` はデフォルトで `goal` == `minimize` です。 |
|  `goal` |  メトリクスの最適化方向を指定します。サポートされている方向: "minimize,maximize" |

| 戻り値 |  |
| :--- | :--- |
|  さらに指定可能なメトリクスオブジェクトが返されます。 |

### `detach`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2848-L2849)

```python
detach() -> None
```

### `display`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1349-L1357)

```python
display(
    height: int = 420,
    hidden: bool = (False)
) -> bool
```

jupyter 内でこの run を表示します。

### `finish`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2086-L2100)

```python
finish(
    exit_code: Optional[int] = None,
    quiet: Optional[bool] = None
) -> None
```

run を終了し、すべてのデータのアップロードを完了します。

これは同じプロセスで複数の run を作成する際に使用されます。スクリプトが終了する際や run コンテキストマネージャを使用する際に、自動的にこのメソッドが呼び出されます。

| 引数 |  |
| :--- | :--- |
|  `exit_code` |  0 以外の値を設定すると run は失敗したと見なされます。 |
|  `quiet` |  ログ出力を最小限にするには true を設定します。 |

### `finish_artifact`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L3098-L3150)

```python
finish_artifact(
    artifact_or_path: Union[Artifact, str],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    distributed_id: Optional[str] = None
) -> Artifact
```

未確定の Artifact を run の出力として完了します。

同じ分散 ID での後続の「アップサート」は新しいバージョンを生成します。

| 引数 |  |
| :--- | :--- |
|  `artifact_or_path` |  (str または Artifact) このアーティファクトの内容へのパス、以下の形式で指定できます: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` また `wandb.Artifact` を呼び出して作成された Artifact オブジェクトを渡すこともできます。 |
|  `name` |  (str、オプション) アーティファクトの名前。エンティティ/プロジェクトで接頭辞を付けることができます。以下の形式で有効です: - name:version - name:alias - digest 指定しない場合は、パスのベース名に現在の run ID が接頭辞として付けられます。 |
|  `type` |  (str) ログするアーティファクトのタイプ、例として `dataset`, `model` など。 |
|  `aliases` |  (リスト、オプション) この Artifact に適用されるエイリアス、デフォルトは `["latest"]`。 |
|  `distributed_id` |  (string、オプション) すべての分散ジョブが共有する一意の文字列。None の場合、run のグループ名がデフォルトです。 |

| 戻り値 |  |
| :--- | :--- |
|  Artifact オブジェクト。 |

### `get_project_url`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1231-L1239)

```python
get_project_url() -> Optional[str]
```

run に関連付けられている W&B プロジェクトの URL を返します（存在する場合）。

オフライン run にはプロジェクト URL はありません。

### `get_sweep_url`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1241-L1246)

```python
get_sweep_url() -> Optional[str]
```

run に関連付けられている sweep の URL を返します（存在する場合）。

### `get_url`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1221-L1229)

```python
get_url() -> Optional[str]
```

run の W&B URL を返します（存在する場合）。

オフライン run には URL はありません。

### `join`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2134-L2144)

```python
join(
    exit_code: Optional[int] = None
) -> None
```

`finish()` の廃止予定のエイリアス - 代わりに finish を使用してください。

### `link_artifact`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2851-L2897)

```python
link_artifact(
    artifact: Artifact,
    target_path: str,
    aliases: Optional[List[str]] = None
) -> None
```

指定されたポートフォリオ（Artifacts の集約コレクション）にアーティファクトをリンクします。

リンクされたアーティファクトは、指定されたポートフォリオの UI 上で表示されます。

| 引数 |  |
| :--- | :--- |
|  `artifact` |  リンクされる（公開またはローカルの）アーティファクト。 |
|  `target_path` |  `str` - 以下の形式で指定してください: {portfolio}, {project}/{portfolio}, または {entity}/{project}/{portfolio} |
|  `aliases` |  `List[str]` - オプションのエイリアス、それはこのポートフォリオ内でリンクされたアーティファクトにのみ適用されます。エイリアス「latest」は常にリンクされたアーティファクトの最新バージョンに適用されます。 |

| 戻り値 |  |
| :--- | :--- |
|  なし |

### `link_model`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L3386-L3478)

```python
link_model(
    path: StrPath,
    registered_model_name: str,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> None
```

モデルアーティファクトバージョンをログし、モデルレジストリに登録されているモデルにリンクします。

リンクされたモデルバージョンは指定された登録モデルの UI に表示されます。

#### 手順:

- 「name」モデルアーティファクトがログされているか確認します。ログされている場合、「path」にあるファイルと一致するアーティファクトバージョンを使用するか、新しいバージョンをログします。そうでない場合、「path」にあるファイルを新しいモデルアーティファクト「name」タイプ「model」としてログします。
- 「registered_model_name」という名前の登録モデルが「model-registry」プロジェクトに存在するか確認します。存在しない場合、「registered_model_name」という名前の新しい登録モデルを作成します。
- モデルアーティファクト「name」のバージョンを登録モデル「registered_model_name」にリンクします。
- 「aliases」リストからのエイリアスを新しくリンクされたモデルアーティファクトバージョンに追加します。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) このモデルの内容へのパス、以下の形式で指定できます: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `registered_model_name` |  (str) - 登録モデルにリンクされるモデルの名前。登録モデルは通常チームの特定の ML タスクを表し、モデルレジストリにリンクされたモデルバージョンのコレクションです。登録モデルが属するエンティティは run 名から派生されます: (str、オプション)

#### Examples:

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
|  `AssertionError` |  registered_model_nameがパスである場合、またはモデルアーティファクトの'name'が'substring'を含まないタイプである場合 |
|  `ValueError` |  nameに無効な特殊文字が含まれている場合 |

| Returns |  |
| :--- | :--- |
|  なし |

### `log`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1665-L1877)

```python
log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None
) -> None
```

現在のrunの履歴にデータの辞書をログします。

`wandb.log`を使用して、スカラー、画像、ビデオ、ヒストグラム、プロット、テーブルなどのデータをログできます。

ライブ例、コードスニペット、ベストプラクティスなどについては、[guides to logging](https://docs.wandb.ai/guides/track/log)をご覧ください。

最も基本的な使用法は `wandb.log({"train-loss": 0.5, "accuracy": 0.9})` です。これは損失と精度をrunの履歴に保存し、これらのメトリクスの概要値を更新します。

ロギングされたデータは、[wandb.ai](https://wandb.ai)のワークスペースで、またはW&Bアプリの[self-hosted instance](https://docs.wandb.ai/guides/hosting)上でローカルに可視化できます。あるいは、データをエクスポートしてローカルで可視化および探索することもできます。たとえば、Jupyterノートブックなどで、[our API](https://docs.wandb.ai/guides/track/public-api-guide)を使用します。

UIでは、概要値がrunテーブルに表示され、run間で単一の値を比較できます。概要値は、`wandb.run.summary["key"] = value` を使って直接設定することもできます。

ログされる値はスカラーである必要はありません。任意の wandb オブジェクトのロギングがサポートされています。たとえば、`wandb.log({"example": wandb.Image("myimage.jpg")})` は例の画像をログし、W&B UIでうまく表示されます。
サポートされているさまざまなタイプについては、[reference documentation](https://docs.wandb.com/ref/python/data-types)をご覧ください。または、[guides to logging](https://docs.wandb.ai/guides/track/log)で、3D分子構造やセグメンテーションマスクからPR曲線やヒストグラムまでの例を確認できます。
`wandb.Table`は構造化データをログするために使用できます。詳細については[guide to logging tables](https://docs.wandb.ai/guides/data-vis/log-tables)をご覧ください。

ネストされたメトリクスのロギングは推奨されており、W&B UIでサポートされています。
`wandb.log({"train": {"acc": 0.9}, "val": {"acc": 0.8}})`のようにネストされた辞書を使ってログすると、メトリクスはW&B UIで `train` と `val` セクションに整理されます。

wandbはグローバルステップを追跡しており、デフォルトでは`wandb.log`の呼び出しごとに増加するため、関連するメトリクスを一緒にログすることが推奨されます。関連するメトリクスを一緒にログするのが難しい場合は、
`wandb.log({"train-loss": 0.5}, commit=False)` を呼び出してから `wandb.log({"accuracy": 0.9})` を呼び出すことは、`wandb.log({"train-loss": 0.5, "accuracy": 0.9})` を呼び出すことと同等です。

`wandb.log`は1秒あたり数回以上呼び出すことを意図していません。それより頻繁にログを取りたい場合は、クライアント側でデータを集約する方が良く、そうしないとパフォーマンスが低下する可能性があります。

| Arguments |  |
| :--- | :--- |
|  `data` |  (辞書, オプション) シリアライズ可能なPythonオブジェクト、すなわち `str`、`int`、`float`、`Tensor`、`辞書` 、または任意の `wandb.data_types` の辞書。 |
|  `commit` |  (ブール値, オプション) メトリクスの辞書をwandbサーバーに保存し、ステップを増加させます。Falseの場合、`wandb.log`は現在のメトリクス辞書を引数のデータで更新するだけで、`commit=True`で`wandb.log`が呼び出されるまでメトリクスは保存されません。 |
|  `step` |  (整数, オプション) プロセッシングのグローバルステップ。これには、以前の未コミットのステップが保持されますが、デフォルトでは指定されたステップはコミットされません。 |
|  `sync` |  (ブール値, True) この引数は廃止されており、現在は `wandb.log` の動作を変更しません。 |

#### Examples:

より詳細な例については、[our guides to logging](https://docs.wandb.com/guides/track/log)をご覧ください。

### 基本的な使用法

```python
import wandb

run = wandb.init()
run.log({"accuracy": 0.9, "epoch": 5})
```

### インクリメンタルロギング

```python
import wandb

run = wandb.init()
run.log({"loss": 0.2}, commit=False)
# 別の場所でこのステップを報告する準備ができたときに:
run.log({"accuracy": 0.8})
```

### ヒストグラム

```python
import numpy as np
import wandb

# 正規分布からランダムにサンプリングされた勾配
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
# 軸は（時間、チャンネル、高さ、幅）
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
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
ax.plot(x, y)  # プロット y = x^2
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

| Raises |  |
| :--- | :--- |
|  `wandb.Error` |  `wandb.init`の前に呼び出された場合 |
|  `ValueError` |  無効なデータが渡された場合 |

### `log_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L3008-L3042)

```python
log_artifact(
    artifact_or_path: Union[Artifact, StrPath],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> Artifact
```

artifactをrunの出力として宣言します。

| Arguments |  |
| :--- | :--- |
|  `artifact_or_path` |  (str または Artifact) このartifactの内容へのパス。次の形式が使用できます: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` また、`wandb.Artifact`を呼び出して作成されたArtifactオブジェクトを渡すこともできます。 |
|  `name` |  (str, オプション) artifact名。次の形式が有効です: - name:version - name:alias - digest 指定されない場合、パスのベース名に現在のrun IDを追加したものがデフォルトになります。 |
|  `type` |  (str) ログするartifactのタイプ。例として`dataset`、`model`などがあります。 |
|  `aliases` |  (リスト, オプション) これに適用するエイリアス。デフォルトは `["latest"]` |

| Returns |  |
| :--- | :--- |
|  An `Artifact` object. |

### `log_code`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1136-L1219)

```python
log_code(
    root: Optional[str] = ".",
    name: Optional[str] = None,
    include_fn: Union[Callable[[str, str], bool], Callable[[str], bool]] = _is_py_or_dockerfile,
    exclude_fn: Union[Callable[[str, str], bool], Callable[[str], bool]] = filenames.exclude_wandb_fn
) -> Optional[Artifact]
```

現在のコードの状態をW&B Artifactに保存します。

デフォルトでは、現在のディレクトリを走査して `.py` で終わるすべてのファイルをログします。

| Arguments |  |
| :--- | :--- |
|  `root` |  コードを再帰的に検索する相対パスまたは絶対パス。`os.getcwd()`からの相対パス。 |
|  `name` |  (str, オプション) コードアーティファクトの名前。デフォルトでは、アーティファクトに`source-$PROJECT_ID-$ENTRYPOINT_RELPATH`という名前が付きます。多数のrunが同じアーティファクトを共有するシナリオがある場合、nameを指定することでそれを実現できます。 |
|  `include_fn` |  ファイルパスと（オプションで）ルートパスを受け取り、それが含まれるべきときにTrueを返し、それ以外の場合にFalseを返す呼び出し可能オブジェクト。 デフォルトでは: `lambda path, root: path.endswith(".py")` |
|  `exclude_fn` |  ファイルパスと（オプションで）ルートパスを受け取り、それが除外されるべきときに`True`を返し、それ以外の場合に`False`を返す呼び出し可能オブジェクト。 デフォルトではルートの`.wandb/`および`wandb/`ディレクトリ内のすべてのファイルを除外する関数です。 |

#### Examples:

基本使用法

```python
run.log_code()
```

高度な使用法

```python
run.log_code(
    "../",
    include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
    exclude_fn=lambda path, root: os.path.relpath(path, root).startswith("cache/"),
)
```

| Returns |  |
| :--- | :--- |
|  コードがログされた場合、An `Artifact` object |

### `log_model`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L3282-L3331)

```python
log_model(
    path: StrPath,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> None
```

runに'path'内の内容を含むモデルアーティファクトをログし、これをこのrunの出力としてマークします。

| Arguments |  |
| :--- | :--- |
|  `path` |  (str) このモデルの内容へのパス。次の形式が使用できます: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `name` |  (str, オプション) ファイル内容が追加されるモデルアーティファクトに割り当てる名前。この文字列には次の英数字の文字のみを含める必要があります: ダッシュ、アンダースコア、およびドット。指定しない場合、パスのベース名に現在のrun IDを追加したものがデフォルトになります。 |
|  `aliases` |  (リスト, オプション) 作成されたモデルアーティファクトに適用するエイリアス。デフォルトは `["latest"]` |

#### Examples:

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
|  `ValueError` |  nameに無効な特殊文字が含まれている場合 |

| Returns |  |
| :--- | :--- |
|  なし |

### `mark_preempting`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L3531-L3539)

```python
mark_preempting() -> None
```

このrunをプリエンプティングとしてマークします。

また、内部プロセスにこれをサーバーに即座に報告するよう指示します。

### `plot_table`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2171-L2192)

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

テーブルでカスタムプロットを作成します。

| Arguments |  |
| :--- | :--- |
|  `vega_spec_name` |  プロットのためのスペックの名前 |
|  `data_table` |  可視化に使用するデータを含むwandb.Tableオブジェクト |
|  `fields` |  テーブルキーからカスタム可視化が必要とするフィールドへのマッピング辞書 |
|  `string_fields` |  カスタム可視化が必要とする任意の文字列定数の値を提供する辞書 |

### `project_name`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1082-L1083)

```python
project_name() -> str
```

### `restore`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2071-L2084)

```python
restore(
    name: str,
    run_path: Optional[str] = None,
    replace: bool = (False),
    root: Optional[str] = None
) -> Union[None, TextIO]
```

指定されたファイルをクラウドストレージからダウンロードします。

ファイルは現在のディレクトリーまたは run のディレクトリーに配置されます。
デフォルトでは、すでに存在しない場合にのみファイルをダウンロードします。

| 引数 |  |
| :--- | :--- |
|  `name` |  ファイルの名前 |
|  `run_path` |  ファイルを取得するための run へのオプションのパス。つまり、 `username/project_name/run_id`。wandb.init が呼び出されていない場合はこれが必要です。 |
|  `replace` |  ファイルがローカルにすでに存在する場合でもダウンロードするかどうか |
|  `root` |  ファイルをダウンロードするディレクトリー。デフォルトでは現在のディレクトリーまたは wandb.init が呼び出された場合の run のディレクトリー。 |

| 戻り値 |  |
| :--- | :--- |
|  ファイルが見つからない場合は None、見つかった場合は読み取り用にオープンされたファイルオブジェクト |

| 例外 |  |
| :--- | :--- |
|  `wandb.CommError` |  wandb バックエンドに接続できない場合 |
|  `ValueError` |  ファイルが見つからない場合や run_path を見つけられない場合 |

### `save`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1879-L1985)

```python
save(
    glob_str: Optional[Union[str, os.PathLike]] = None,
    base_path: Optional[Union[str, os.PathLike]] = None,
    policy: "PolicyName" = "live"
) -> Union[bool, List[str]]
```

1つまたは複数のファイルを W&B に同期します。

相対パスは現在の作業ディレクトリーに対して相対的に解釈されます。

Unix グロブパターン（例: "myfiles/*"）は、`save` が呼び出された時点で展開され、`policy` の設定にかかわらず処理されます。特に、新しいファイルは自動的にピックアップされません。

`base_path` を指定することでアップロードファイルのディレクトリー構造を制御できます。それは `glob_str` のプレフィックスであり、その下のディレクトリー構造は維持されます。以下の例で理解できるでしょう：

```
wandb.save("these/are/myfiles/*")
# => ファイルは run の "these/are/myfiles/" フォルダーに保存されます。

wandb.save("these/are/myfiles/*", base_path="these")
# => ファイルは run の "are/myfiles/" フォルダーに保存されます。

wandb.save("/User/username/Documents/run123/*.txt")
# => ファイルは run の "run123/" フォルダーに保存されます。以下の注意を参照。

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => ファイルは run の "username/Documents/run123/" フォルダーに保存されます。

wandb.save("files/*/saveme.txt")
# => 各 "saveme.txt" ファイルは "files/" の適切なサブディレクトリーに保存されます。
```

注意: 絶対パスやグロブが `base_path` なしで指定された場合、上記の例のように1つのディレクトリーレベルが維持されます。

| 引数 |  |
| :--- | :--- |
|  `glob_str` |  相対または絶対パスまたは Unix グロブパターン。 |
|  `base_path` |  ディレクトリー構造を推測するためのパス；例を参照。 |
|  `policy` |  `live`, `now`, または `end` のいずれか。 * live: 変更されるたびにファイルをアップロードし、前のバージョンを上書きします * now: 現在のファイルを一度だけアップロードします * end: run が終了したときにファイルをアップロードします |

| 戻り値 |  |
| :--- | :--- |
|  マッチするファイルのシンボリックリンクが作成されたパス。歴史的な理由で、レガシーコードではブール値を返すことがあります。 |

### `status`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2146-L2169)

```python
status() -> RunStatus
```

現在の run の同期状態に関する内部バックエンドの情報を取得します。

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1359-L1368)

```python
to_html(
    height: int = 420,
    hidden: bool = (False)
) -> str
```

現在の run を表示する iframe を含む HTML を生成します。

### `unwatch`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2809-L2811)

```python
unwatch(
    models=None
) -> None
```

### `upsert_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L3044-L3096)

```python
upsert_artifact(
    artifact_or_path: Union[Artifact, str],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    distributed_id: Optional[str] = None
) -> Artifact
```

未完了のアーティファクトを run の出力として宣言（または追加）します。

run.finish_artifact() を呼び出してアーティファクトを完了させる必要があります。
これは、分散ジョブが同じアーティファクトにすべて貢献する必要がある場合に便利です。

| 引数 |  |
| :--- | :--- |
|  `artifact_or_path` |  (str または Artifact) このアーティファクトの内容へのパス、以下の形式のいずれかです: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` また、`wandb.Artifact` を呼び出して作成された Artifact オブジェクトを渡すこともできます。 |
|  `name` |  (str, オプション) アーティファクトの名前。entity/project でプレフィクスすることができます。有効な名前は次の形式である必要があります: - name:version - name:alias - digest 指定しない場合、デフォルトではパスのベース名が現在の run ID によって補われます。 |
|  `type` |  (str) ログするアーティファクトの種類、例：`dataset`、`model` |
|  `aliases` |  (リスト, オプション) このアーティファクトに適用するエイリアス、デフォルトは `["latest"]` |
|  `distributed_id` |  (string, オプション) すべての分散ジョブが共有する一意の文字列。None の場合、run のグループ名がデフォルトとして使用されます。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `use_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2899-L3006)

```python
use_artifact(
    artifact_or_name: Union[str, Artifact],
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    use_as: Optional[str] = None
) -> Artifact
```

アーティファクトを run の入力として宣言します。

戻り値のオブジェクトに対して `download` または `file` を呼び出してローカルに内容を取得します。

| 引数 |  |
| :--- | :--- |
|  `artifact_or_name` |  (str または Artifact) アーティファクトの名前。entity/project でプレフィクスすることができます。有効な名前は次の形式である必要があります: - name:version - name:alias また、`wandb.Artifact` を呼び出して作成された Artifact オブジェクトを渡すこともできます。 |
|  `type` |  (str, オプション) 使用するアーティファクトの種類。 |
|  `aliases` |  (リスト, オプション) このアーティファクトに適用するエイリアス |
|  `use_as` |  (文字列, オプション) アーティファクトがどのような目的で使用されたかを示すオプションの文字列。UI に表示されます。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `use_model`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L3333-L3384)

```python
use_model(
    name: str
) -> FilePathStr
```

モデルアーティファクト 'name' にログされたファイルをダウンロードします。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) モデルアーティファクトの名前。'name' は既存のログされたモデルアーティファクトの名前と一致しなければなりません。entity/project でプレフィクスすることができます。有効な名前は次の形式である必要があります: - model_artifact_name:version - model_artifact_name:alias |

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
|  `AssertionError` |  モデルアーティファクト 'name' が 'model' というサブストリングを含まないタイプの場合。 |

| 戻り値 |  |
| :--- | :--- |
|  `path` |  (str) ダウンロードされたモデルアーティファクトファイルのパス。 |

### `watch`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2796-L2806)

```python
watch(
    models, criterion=None, log="gradients", log_freq=100, idx=None,
    log_graph=(False)
) -> None
```

### `__enter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L3515-L3516)

```python
__enter__() -> "Run"
```

### `__exit__`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L3518-L3529)

```python
__exit__(
    exc_type: Type[BaseException],
    exc_val: BaseException,
    exc_tb: TracebackType
) -> bool
```