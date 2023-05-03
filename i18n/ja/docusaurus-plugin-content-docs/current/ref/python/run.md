# Run

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L405-L3638)

wandbによってログされる計算の単位。通常、これはML実験です。

```python
Run(
 settings: Settings,
 config: Optional[Dict[str, Any]] = None,
 sweep_config: Optional[Dict[str, Any]] = None,
 launch_config: Optional[Dict[str, Any]] = None
) -> None
```

`wandb.init()`を使ってrunを作成します:

```python
import wandb

run = wandb.init()
```

任意のプロセスでアクティブな `wandb.Run` は最大で1つだけであり、`wandb.run` としてアクセスできます:

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```
`wandb.log` でログしたものは、そのrunに送られます。
同じスクリプトやノートブックで複数のrunを開始したい場合は、進行中のrunを終了する必要があります。runは`wandb.finish`で終了させるか、`with`ブロックを使用して終了させます。

```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
 pass # ここでデータをログに記録

assert wandb.run is None
```

runの作成については`wandb.init`のドキュメントを参照してください。[`wandb.init`に関する当社のガイド](https://docs.wandb.ai/guides/track/launch)もチェックしてみてください。

分散トレーニングでは、ランク0プロセスで単一のrunを作成し、そのプロセスからの情報のみをログに記録するか、各プロセスでrunを作成し、それぞれ別々にログに記録し、`wandb.init`の`group`引数で結果をまとめることができます。W&Bを使った分散トレーニングの詳細については、[当社のガイド](https://docs.wandb.ai/guides/track/log/distributed-training)をご覧ください。

現在、`wandb.Api`には並列`Run`オブジェクトが存在しています。これら2つのオブジェクトは最終的には統合される予定です。

| 属性 | |
| :--- | :--- |
| `summary` | (Summary) 各`wandb.log()`キーに設定された単一の値。デフォルトでは、サマリーは最後にログされた値に設定されます。サマリーを最後の値ではなく、最大精度などの最適な値に手動で設定することができます。 |
| `config` | このrunに関連する設定オブジェクト。 |
| `dir` | runに関連するファイルが保存されているディレクトリ。 |
| `entity` | runに関連するW&Bエンティティの名前。エンティティはユーザー名やチーム名、組織名などになります。 |
| `group` | runに関連するグループ名。グループを設定すると、W&BのUIがrunを適切な方法で整理するのに役立ちます。分散トレーニングを行っている場合は、トレーニング中のすべてのランに同じグループ名を付ける必要があります。クロスバリデーションを行っている場合は、すべてのクロスバリデーションフォールドに同じグループ名を付ける必要があります。 |
| `id` | このrunの識別子。 |
| `mode` | `0.9.x`以前の互換性のために、最終的に非推奨。 |
| `name` | runの表示名。表示名は一意であることが保証されておらず、説明的であることがあります。デフォルトでは、ランダムに生成されます。 |
| `notes` | runに関連するメモがあれば、そのメモ。メモは複数行の文字列で、マークダウンや`$$`内のlatex方程式（`$x + 3$`のような）も使用できます。 |
| `path` | runへのパス。runパスにはエンティティ、プロジェクト、およびrun IDが含まれ、`entity/project/run_id`の形式で表示されます。 |
| `project` | runに関連するW&Bプロジェクトの名前。 |
| `resumed` | runが再開された場合はTrue、そうでない場合はFalse。 |
| `settings` | runの設定オブジェクトの凍結コピー。 |
| `start_time` | runが開始された時のUnixタイムスタンプ（秒単位）。 |
| `starting_step` | runの最初のステップ。 |
| `step` | ステップの現在の値。このカウンターは`wandb.log`によってインクリメントされます。 |
| `sweep_id` | runに関連するスイープのID（ある場合）。 |
| `tags` | runに関連するタグがあれば、それらのタグ。 |
| `url` | runに関連するW&Bのurl。 |
## メソッド

### `alert`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L3007-L3040)

```python
alert(
 title: str,
 text: str,
 level: Optional[Union[str, 'AlertLevel']] = None,
 wait_duration: Union[int, float, timedelta, None] = None
) -> None
```

指定されたタイトルとテキストでアラートを起動します。


| 引数 | |
| :--- | :--- |
| `title` | (str) アラートのタイトル。64文字未満である必要があります。 |
| `text` | (str) アラートの本文。 |
| `level` | (strまたはwandb.AlertLevel、オプション) 使用するアラートレベル。 `INFO`、`WARN`、または `ERROR` のいずれか。 |
| `wait_duration` | (int、float、timedelta、オプション) このタイトルで他のアラートを送信するまでの待機時間（秒）。 |



### `define_metric`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L2426-L2460)

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
`wandb.log()`で後でログに記録されるメトリックのプロパティを定義します。

| 引数 | |
| :--- | :--- |
| `name` | メトリックの名前。 |
| `step_metric` | メトリックに関連する独立変数。 |
| `step_sync` | 必要に応じて`step_metric`を履歴に自動追加。`step_metric`が指定されている場合、デフォルトでTrue。 |
| `hidden` | このメトリックを自動プロットから非表示にします。 |
| `summary` | サマリーに追加される集計メトリックを指定。サポートされる集計: "min,max,mean,best,last,none" デフォルトの集計は`copy` 集計`best`はデフォルトで`goal`==`minimize` |
| `goal` | メトリックの最適化方向を指定します。サポートされる方向: "minimize,maximize" |



| 戻り値 | |
| :--- | :--- |
| さらに指定可能なメトリックオブジェクトが返されます。 |



### `detach`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L2586-L2587)

```python
detach() -> None
```




### `display`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1252-L1260)

```python
display(
 height: int = 420,
 hidden: bool = (False)
) -> bool
```
この実行をjupyterで表示します。

### `finish`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1869-L1883)

```python
finish(
 exit_code: Optional[int] = None,
 quiet: Optional[bool] = None
) -> None
```

実行を終了済みとしてマークし、すべてのデータのアップロードを完了します。

これは、同じプロセスで複数のrunを作成する場合に使用されます。スクリプトが終了するか、runコンテキストマネージャを使用した場合に、このメソッドを自動的に呼び出します。

| 引数 | |
| :--- | :--- |
| `exit_code` | 実行を失敗としてマークするには、0以外の値を設定します |
| `quiet` | ログ出力を最小限に抑えるには、trueに設定します |



### `finish_artifact`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L2832-L2884)

```python
finish_artifact(
 artifact_or_path: Union[wandb_artifacts.Artifact, str],
 name: Optional[str] = None,
 type: Optional[str] = None,
 aliases: Optional[List[str]] = None,
 distributed_id: Optional[str] = None
) -> wandb_artifacts.Artifact
```
runの出力として、非確定的なアーティファクトを終了させます。

同じ分散IDでの後続の "upserts" は、新しいバージョンが生成されます。

| 引数 | |
| :--- | :--- |
| `artifact_or_path` | (str または Artifact) このアーティファクトの内容へのパスで、以下の形式があります： - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` `wandb.Artifact` を呼び出すことで作成される Artifact オブジェクトも渡すことができます。 |
| `name` | (str, 任意) アーティファクト名。エンティティ/プロジェクトでプリフィックス付きも可。有効な名前は以下の形式であることができます: - name:version - name:alias - digest 指定されていない場合、パスのベース名に現在のrun IDが追加されたものがデフォルトになります。 |
| `type` | (str) ログに残すアーティファクトの種類。例えば、`dataset` や `model` など |
| `aliases` | (list, 任意) このアーティファクトに適用するエイリアス。デフォルトは `["latest"]` です |
| `distributed_id` | (string, 任意) すべての分散ジョブが共有するユニークな文字列。None の場合、runのグループ名がデフォルトになります。 |



| 戻り値 | |
| :--- | :--- |
| `Artifact`オブジェクト。 |



### `get_project_url`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1132-L1140)

```python
get_project_url() -> Optional[str]
```

runに関連付けられたW&BプロジェクトのURLを返します（ある場合）。

オフラインのrunでは、プロジェクトのURLはありません。

### `get_sweep_url`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1142-L1147)
以下は、Markdownテキストのチャンクを翻訳してください。日本語に翻訳し、それ以外のことは言わずに翻訳したテキストのみを返してください。テキスト：

```python
get_sweep_url() -> Optional[str]
```

runに関連付けられたスイープのurlを返します（もしあれば）。

### `get_url`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1122-L1130)

```python
get_url() -> Optional[str]
```

W&B runのurlを返します（もしあれば）。

オフラインのrunではurlがありません。

### `join`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1917-L1927)

```python
join(
 exit_code: Optional[int] = None
) -> None
```

`finish()`の非推奨エイリアス - 代わりにfinishを使用してください。

### `link_artifact`
[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L2589-L2631)

```python
link_artifact(
 artifact: Union[public.Artifact, Artifact],
 target_path: str,
 aliases: Optional[List[str]] = None
) -> None
```

指定されたアーティファクトをポートフォリオ（プロモートされたアーティファクトのコレクション）にリンクします。

リンクされたアーティファクトは、指定されたポートフォリオのUIで表示されます。

| 引数 | |
| :--- | :--- |
| `artifact` | リンクされる（パブリックまたはローカル）アーティファクト |
| `target_path` | `str` - 次の形式をとります：{portfolio}、{project}/{portfolio}、または {entity}/{project}/{portfolio} |
| `aliases` | `List[str]` - ポートフォリオ内のこのリンクされたアーティファクトに適用されるオプションのエイリアス。エイリアス "latest" は、リンクされたアーティファクトの最新バージョンに常に適用されます。 |



| 戻り値 | |
| :--- | :--- |
| None |



### `log`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1555-L1750)

```python
log(
 data: Dict[str, Any],
 step: Optional[int] = None,
 commit: Optional[bool] = None,
 sync: Optional[bool] = None
) -> None
```
現在のrunの履歴にデータの辞書を記録します。

`wandb.log`を使用して、スカラ、画像、ビデオ、ヒストグラム、プロット、テーブルなどのrunからデータを記録します。

ライブの例、コードスニペット、ベストプラクティスなどを知るには、[記録のガイド](https://docs.wandb.ai/guides/track/log)を参照してください。

最も基本的な使い方は `wandb.log({"train-loss": 0.5, "accuracy": 0.9})` です。これにより、損失と精度がrunの履歴に保存され、これらのメトリックのサマリー値が更新されます。

記録されたデータは、[wandb.ai](https://wandb.ai) のワークスペースで可視化したり、W&Bアプリの[セルフホスティングされたインスタンス](https://docs.wandb.ai/guides/hosting)上でローカルに表示したり、[API](https://docs.wandb.ai/guides/track/public-api-guide)を使用してJupyterノートブックなどでローカルに可視化・探索したりできます。

UIでは、サマリー値は、run間で単一の値を比較するためのrunテーブルに表示されます。サマリー値は、`wandb.run.summary["key"] = value`で直接設定することもできます。

記録する値はスカラである必要はありません。wandbオブジェクトの記録がサポートされています。例えば、`wandb.log({"example": wandb.Image("myimage.jpg")})`では、W&BのUIで適切に表示される例の画像を記録します。サポートされているさまざまなタイプについては、[リファレンスドキュメント](https://docs.wandb.com/ref/python/data-types)を参照するか、3D分子構造やセグメンテーションマスク、PR曲線、ヒストグラムなどを含む例が掲載されている[記録のガイド](https://docs.wandb.ai/guides/track/log)をご覧ください。`wandb.Table`は構造化データを記録するために使用できます。詳細については[テーブルの記録ガイド](https://docs.wandb.ai/guides/data-vis/log-tables)を参照してください。

入れ子になったメトリクスの記録が推奨されており、W&BのUIでサポートされています。入れ子になった辞書で記録する場合、`wandb.log({"train": {"acc": 0.9}, "val": {"acc": 0.8}})`とすると、W&BのUIではメトリックが`train`と`val`のセクションに整理されます。

wandbはグローバルなステップを維持しており、デフォルトでは`wandb.log`を呼び出すたびにインクリメントされます。そのため、関連するメトリクスをまとめて記録することが推奨されます。関連するメトリクスをまとめて記録するのが不便な場合は、`wandb.log({"train-loss": 0.5}, commit=False)` とした後に `wandb.log({"accuracy": 0.9})` と呼び出すことは、`wandb.log({"train-loss": 0.5, "accuracy": 0.9})` を呼び出すことと同等です。
`wandb.log`は、1秒あたり数回以上呼び出されることを想定していません。
もし、それより頻繁にログを取りたい場合は、クライアント側でデータを集約するか、
パフォーマンスが低下することがあります。

| 引数 | |
| :--- | :--- |
| `data` | (辞書, 任意) シリアライズ可能なPythonオブジェクトの辞書です。`str`、`ints`、`floats`、`Tensors`、`dicts`、または`wandb.data_types`のいずれか。 |
| `commit` | (boolean, 任意) メトリクスの辞書をwandbサーバーに保存し、ステップをインクリメントします。`false`の場合、`wandb.log`はdata引数で現在のメトリクス辞書を更新し、`commit=True`で`wandb.log`が呼び出されるまでメトリクスは保存されません。 |
| `step` | (整数, 任意) 処理のグローバルステップです。これにより、指定されたステップのコミットをデフォルトで無効にする以前の非コミット済みステップが保持されます。 |
| `sync` | (boolean, True) この引数は非推奨であり、現在は`wandb.log`の振る舞いは変更されません。 |



#### 例:

より多くの詳細な例については、
[ログの取り方に関するガイド](https://docs.wandb.com/guides/track/log)を参照してください。

### 基本的な使い方

```python
import wandb

wandb.init()
wandb.log({"accuracy": 0.9, "epoch": 5})
```

### インクリメンタルなログ

```python
import wandb

wandb.init()
wandb.log({"loss": 0.2}, commit=False)
# どこか別の場所で、このステップを報告する準備ができたとき:
wandb.log({"accuracy": 0.8})
```

### ヒストグラム
以下はMarkdownテキストの日本語訳です。他の言葉を使わず、翻訳されたテキストのみを返してください。テキスト:

```python
import numpy as np
import wandb

# 正規分布からランダムに勾配をサンプリングします
gradients = np.random.randn(100, 100)
wandb.init()
wandb.log({"gradients": wandb.Histogram(gradients)})
```

### numpyからの画像

```python
import numpy as np
import wandb

wandb.init()
examples = []
for i in range(3):
 pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
 image = wandb.Image(pixels, caption=f"random field {i}")
 examples.append(image)
wandb.log({"examples": examples})
```

### PILからの画像

```python
import numpy as np
from PIL import Image as PILImage
import wandb

wandb.init()
examples = []
for i in range(3):
 pixels = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
 pil_image = PILImage.fromarray(pixels, mode="RGB")
 image = wandb.Image(pil_image, caption=f"random field {i}")
 examples.append(image)
wandb.log({"examples": examples})
```
### numpyからのビデオ

```python
import numpy as np
import wandb

wandb.init()
# 軸は (時間, チャンネル, 高さ, 幅)
frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
wandb.log({"video": wandb.Video(frames, fps=4)})
```

### Matplotlibプロット

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

wandb.init()
fig, ax = plt.subplots()
x = np.linspace(0, 10)
y = x * x
ax.plot(x, y) # y = x^2のプロット
wandb.log({"chart": fig})
```

### PR曲線
```python
wandb.log({"pr": wandb.plots.precision_recall(y_test, y_probas, labels)})
```

### 3Dオブジェクト
```python
wandb.log(
 {
 "generated_samples": [
 wandb.Object3D(open("sample.obj")),
 wandb.Object3D(open("sample.gltf")),
 wandb.Object3D(open("sample.glb")),
 ]
 }
)
```
以下のマークダウンテキストを日本語に翻訳してください。翻訳したテキストだけを返し、それ以外のことは何も言わないでください。テキスト:

| 例外 | |
| :--- | :--- |
| `wandb.Error` | `wandb.init`の前に呼び出された場合 |
| `ValueError` | 不正なデータが渡された場合 |



### `log_artifact`



[ソースコードを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L2741-L2776)

```python
log_artifact(
 artifact_or_path: Union[wandb_artifacts.Artifact, StrPath],
 name: Optional[str] = None,
 type: Optional[str] = None,
 aliases: Optional[List[str]] = None
) -> wandb_artifacts.Artifact
```

アーティファクトをrunの出力として宣言します。


| 引数 | |
| :--- | :--- |
| `artifact_or_path` | (str または Artifact) このアーティファクトの内容へのパスで、次の形式で表すことができます: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` また、`wandb.Artifact`を呼び出して作成されたArtifactオブジェクトを渡すこともできます。 |
| `name` | (str, オプション) アーティファクト名。エンティティ/プロジェクトでプレフィックスが付けられている場合もあります。有効な名前は次の形式で表すことができます: - name:version - name:alias - digest 指定されていない場合、デフォルトではパスのベース名が現在のrun IDに前置されます。 |
| `type` | (str) ログするアーティファクトの種類。`dataset` や `model` などの例があります。 |
| `aliases` | (list, オプション) このアーティファクトに適用するエイリアスで、デフォルトは `["latest"]` |



| 返り値 | |
| :--- | :--- |
| `Artifact`オブジェクト。 |
### `log_code`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1057-L1120)

```python
log_code(
 root: Optional[str] = ".",
 name: Optional[str] = None,
 include_fn: Callable[[str], bool] = _is_py_path,
 exclude_fn: Callable[[str], bool] = filenames.exclude_wandb_fn
) -> Optional[Artifact]
```

現在のコードの状態をW&Bアーティファクトに保存します。

デフォルトでは、現在のディレクトリーを走査し、`.py`で終わるすべてのファイルをログします。

| 引数 | |
| :--- | :--- |
| `root` | コードを再帰的に検索する相対( `os.getcwd()` に対する)または絶対パス。 |
| `name` | (str, オプション) コードアーティファクトの名前。デフォルトでは、アーティファクトの名前は `source-$PROJECT_ID-$ENTRYPOINT_RELPATH` になります。複数のrunが同じアーティファクトを共有するシナリオがある場合、nameを指定することで実現できます。 |
| `include_fn` | ファイルパスを受け入れ、含めるべき場合は `True` を、そうでない場合は `False` を返す呼び出し可能なオブジェクト。デフォルトでは、`lambda path: path.endswith(".py")` になります。 |
| `exclude_fn` | ファイルパスを受け入れ、除外すべき場合は `True` を、そうでない場合は `False` を返す呼び出し可能なオブジェクト。デフォルトでは、`lambda path: False` になります。 |



#### 例:

基本的な使い方
```python
run.log_code()
```

高度な使い方
```python
run.log_code(
 "../", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb")
)
```
| Returns | |
| :--- | :--- |
| コードがログされた場合、`Artifact`オブジェクト |



### `mark_preempting`



[ソースを表示する](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L3055-L3063)

```python
mark_preempting() -> None
```

このrunを事前停止としてマークします。

また、内部プロセスに対して、これをすぐにサーバーに報告するよう指示します。

### `plot_table`



[ソースを表示する](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1954-L1972)

```python
@staticmethod
plot_table(
 vega_spec_name: str,
 data_table: "wandb.Table",
 fields: Dict[str, Any],
 string_fields: Optional[Dict[str, Any]] = None
) -> CustomChart
```

テーブル上にカスタムプロットを作成します。
| 引数 | |
| :--- | :--- |
| `vega_spec_name` | プロットの仕様名 |
| `data_table` | 可視化で使用されるデータを含むwandb.Tableオブジェクト |
| `fields` | カスタム可視化が必要とするフィールドへのテーブルキーのマッピングを表す辞書 |
| `string_fields` | カスタム可視化が必要とする文字列定数の値を提供する辞書 |



### `project_name`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1000-L1002)

```python
project_name() -> str
```




### `restore`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1854-L1867)

```python
restore(
 name: str,
 run_path: Optional[str] = None,
 replace: bool = (False),
 root: Optional[str] = None
) -> Union[None, TextIO]
```

指定されたファイルをクラウドストレージからダウンロードします。
ファイルは現在のディレクトリーまたはrunディレクトリーに配置されます。
デフォルトでは、ファイルがすでに存在しない場合にのみダウンロードします。

| 引数 |  |
| :--- | :--- |
| `name` | ファイルの名前 |
| `run_path` | ファイルを取得するrunへのオプションのパス。例：`username/project_name/run_id` 。wandb.initが呼び出されていない場合、この引数は必須です。 |
| `replace` | ファイルがローカルに既に存在していてもダウンロードするかどうか |
| `root` | ファイルをダウンロードするディレクトリー。デフォルトでは、現在のディレクトリーまたは、wandb.initが呼び出されている場合はrunディレクトリーになります。 |



| 戻り値 |  |
| :--- | :--- |
| ファイルが見つからない場合はNone、それ以外の場合は読み取り用に開いたファイルオブジェクト |



| 例外 |  |
| :--- | :--- |
| `wandb.CommError` | wandbのバックエンドに接続できない場合 |
| `ValueError` | ファイルが見つからないか、run_pathが見つからない場合 |



### `save`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1752-L1782)

```python
save(
 glob_str: Optional[str] = None,
 base_path: Optional[str] = None,
 policy: "PolicyName" = "live"
) -> Union[bool, List[str]]
```

指定されたポリシーで`glob_str`に一致するすべてのファイルがwandbと同期されることを保証します。
| 引数 | |
| :--- | :--- |
| `glob_str` | (string) unix globまたは正規パスへの相対または絶対パス。これが指定されていない場合、メソッドは何もしません。 |
| `base_path` | (string) globを実行するための基本パス |
| `policy` | (string) `live`、`now`、または `end` のうちの1つ - live: ファイルを変更するたびにアップロードし、前のバージョンを上書きします - now: ファイルを今すぐ一度アップロードします - end: ランが終了するときだけファイルをアップロードします |



### `status`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1929-L1952)

```python
status() -> RunStatus
```

内部バックエンドから現在のランの同期状況に関する情報を取得します。


### `to_html`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1262-L1271)

```python
to_html(
 height: int = 420,
 hidden: bool = (False)
) -> str
```

現在のランを表示するiframeを含むHTMLを生成します。


### `unwatch`
以下はMarkdownテキストのチャンクを翻訳してください。それを日本語に翻訳してください。他のことは何も言わずに、翻訳したテキストだけを返してください。テキスト：

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L2547-L2549)

```python
unwatch(
 models=None
) -> None
```




### `upsert_artifact`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L2778-L2830)

```python
upsert_artifact(
 artifact_or_path: Union[wandb_artifacts.Artifact, str],
 name: Optional[str] = None,
 type: Optional[str] = None,
 aliases: Optional[List[str]] = None,
 distributed_id: Optional[str] = None
) -> wandb_artifacts.Artifact
```

実行の出力として非確定アーティファクトを宣言（または追加）します。

run.finish_artifact()を呼び出してアーティファクトを確定させる必要があることに注意してください。
これは、分散ジョブがすべて同じアーティファクトに貢献する必要がある場合に便利です。

| 引数 | |
| :--- | :--- |
| `artifact_or_path` | (str または Artifact) このアーティファクトの内容へのパス。次の形式であることができます：- `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` または、`wandb.Artifact`を呼び出すことで作成されたArtifactオブジェクトを渡すこともできます。 |
| `name` | (str, 任意) アーティファクト名。エンティティ/プロジェクトでプレフィックスが付けられる場合があります。有効な名前は次の形式であることができます：- name:version - name:alias - digest 指定されていない場合、これはデフォルトでパスのベース名に現在の実行IDが付加されます。|
| `type` | (str) ログに記録するアーティファクトのタイプ。例えば `dataset`、`model` など|
| `aliases` | (リスト, 任意) このアーティファクトに適用するエイリアス。デフォルトでは `["latest"]` になります。|
| `distributed_id` | (string, 任意) すべての分散ジョブが共有する一意の文字列。Noneの場合、実行のグループ名がデフォルトで使用されます。|
| Returns | |
| :--- | :--- |
| `Artifact`オブジェクト。 |



### `use_artifact`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L2633-L2739)

```python
use_artifact(
 artifact_or_name: Union[str, public.Artifact, Artifact],
 type: Optional[str] = None,
 aliases: Optional[List[str]] = None,
 use_as: Optional[str] = None
) -> Union[public.Artifact, Artifact]
```

アーティファクトをrunの入力として宣言します。

返されたオブジェクトで`download`や`file`を呼び出して、ローカルに内容を取得します。

| 引数 | |
| :--- | :--- |
| `artifact_or_name` | (strまたはArtifact) アーティファクト名。エンティティ/プロジェクト/で接頭辞付け可能です。有効な名前は以下の形式があります: - name:version - name:alias - digest また、`wandb.Artifact`を呼び出して作成されたArtifactオブジェクトを渡すこともできます。 |
| `type` | (str, 任意) 使用するアーティファクトのタイプ。 |
| `aliases` | (list, 任意) このアーティファクトに適用するエイリアス |
| `use_as` | (string, 任意) このアーティファクトがどの目的で使用されたかを示す任意の文字列。UIに表示されます。 |



| Returns | |
| :--- | :--- |
| `Artifact`オブジェクト。 |
### `watch`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L2534-L2544)

```python
watch(
 モデル, criterion=None, ログ="gradients", log_freq=100, idx=None,
 log_graph=(False)
) -> None
```




### `__enter__`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L3042-L3043)

```python
__enter__() -> "Run"
```




### `__exit__`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L3045-L3053)

```python
__exit__(
 exc_type: Type[BaseException],
 exc_val: BaseException,
 exc_tb: TracebackType
) -> bool
```