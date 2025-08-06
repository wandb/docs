---
title: Run
object_type: python_sdk_actions
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py >}}




## <kbd>class</kbd> `Run`
W&B で記録される計算の単位です。通常は ML 実験を指します。

新しい run を作成するには [`wandb.init()`](https://docs.wandb.ai/ref/python/init/) を呼び出してください。`wandb.init()` は新たな run を開始し、`wandb.Run` オブジェクトを返します。各 run にはユニークな ID（run ID）が付与されます。W&B ではコンテキストマネージャ（`with` 文）を使って、自動的に run を終了させることを推奨しています。

分散トレーニング実験の場合、プロセスごとに個別の run でトラッキングすることも、全プロセスを1つの run でまとめてトラッキングすることもできます。詳細は [Log distributed training experiments](https://docs.wandb.ai/guides/track/log/distributed-training) をご覧ください。

`wandb.Run.log()` を使って run にデータを記録できます。`wandb.Run.log()` で記録する全ての情報はその run に送信されます。詳細は [Create an experiment](https://docs.wandb.ai/guides/track/launch) または [`wandb.init`](https://docs.wandb.ai/ref/python/init/) APIリファレンスをご覧ください。

また、[`wandb.apis.public`](https://docs.wandb.ai/ref/python/public-api/api/) 名前空間にも別の `Run` オブジェクトがあります。こちらは既に作成された runs とやり取りする際に利用します。



**属性:**
 
 - `summary`:  (Summary) run の概要となる辞書型オブジェクト。詳細は [Log summary metrics](https://docs.wandb.ai/guides/track/log/log-summary/) をご覧ください。



**例:**
`wandb.init()` で run を作成:

```python
import wandb

# 新しい run を開始してデータを記録
# コンテキストマネージャ（with文）で自動的に run を終了
with wandb.init(entity="entity", project="project") as run:
    run.log({"accuracy": acc, "loss": loss})
``` 


### <kbd>property</kbd> Run.config

この run に紐づく config オブジェクト。

---

### <kbd>property</kbd> Run.config_static

この run に紐づく静的 config オブジェクト。

---

### <kbd>property</kbd> Run.dir

run に関連するファイルが保存されるディレクトリー。

---

### <kbd>property</kbd> Run.disabled

この run が無効なら True、そうでなければ False。

---

### <kbd>property</kbd> Run.entity

この run に紐づく W&B entity 名。

entity はユーザー名、チーム名、もしくは組織名になります。

---

### <kbd>property</kbd> Run.group

この run に紐づくグループ名を返します。

runs をグループ化すると、関連する実験を W&B UI でまとめて整理・可視化できます。特に分散トレーニングやクロスバリデーションなど、複数の runs を1つの実験として管理する場合に便利です。

全プロセスが同じ run オブジェクトを共有する shared モードでは、グループ設定は通常不要です（run が1つなのでグループ化が不要なため）。

---

### <kbd>property</kbd> Run.id

この run の ID。

---

### <kbd>property</kbd> Run.job_type

run に紐づくジョブタイプ名。

run のジョブタイプは W&B アプリの run 概要ページで確認できます。

ジョブタイプ（例: "training", "evaluation", "inference" など）で runs を分類するのに便利です。同じプロジェクト内で複数の種類の runs がある場合、W&B UI で管理・絞り込みしやすくなります。詳細は [Organize runs](https://docs.wandb.ai/guides/runs/#organize-runs) を参照してください。

---

### <kbd>property</kbd> Run.name

run の表示名。

表示名は必ずしも一意ではなく、説明的な名前も利用できます。デフォルトではランダムに生成されます。

---

### <kbd>property</kbd> Run.notes

run に付随するノート（あれば）。

複数行の文字列や、マークダウン・LaTeX数式（`$$` で囲む）も利用可能です。例: `$x + 3$`

---

### <kbd>property</kbd> Run.offline

この run がオフラインの場合 True、そうでなければ False。

---

### <kbd>property</kbd> Run.path

run のパス。

パス形式は `entity/project/run_id` です。

---

### <kbd>property</kbd> Run.project

この run に紐付いた W&B プロジェクト名。

---

### <kbd>property</kbd> Run.project_url

この run に紐付いた W&B プロジェクトの URL（あれば）。

オフライン run には project URL はありません。

---

### <kbd>property</kbd> Run.resumed

run が再開された場合は True、そうでなければ False。

---

### <kbd>property</kbd> Run.settings

run の Settings オブジェクトの凍結コピー。

---

### <kbd>property</kbd> Run.start_time

run が開始した Unix タイムスタンプ（秒単位）。

---


### <kbd>property</kbd> Run.sweep_id

この run に関連する sweep の ID（あれば）。

---

### <kbd>property</kbd> Run.sweep_url

この run に関連する sweep の URL（あれば）。

オフライン run には sweep URL はありません。

---

### <kbd>property</kbd> Run.tags

この run に紐づくタグ（あれば）。

---

### <kbd>property</kbd> Run.url

この W&B run の URL（あれば）。

オフライン run には URL がありません。



---

### <kbd>method</kbd> `Run.alert`

```python
alert(
    title: 'str',
    text: 'str',
    level: 'str | AlertLevel | None' = None,
    wait_duration: 'int | float | timedelta | None' = None
) → None
```

指定したタイトルと本文でアラートを作成します。



**引数:**
 
 - `title`:  アラートのタイトル。64文字以内。 
 - `text`:  アラート本文。 
 - `level`:  アラートレベル。`INFO`、`WARN`、`ERROR` のいずれか。 
 - `wait_duration`:  同じタイトルのアラートを再度送信するまで待つ秒数。

---

### <kbd>method</kbd> `Run.define_metric`

```python
define_metric(
    name: 'str',
    step_metric: 'str | wandb_metric.Metric | None' = None,
    step_sync: 'bool | None' = None,
    hidden: 'bool | None' = None,
    summary: 'str | None' = None,
    goal: 'str | None' = None,
    overwrite: 'bool | None' = None
) → wandb_metric.Metric
```

`wandb.Run.log()` で記録されるメトリクスのカスタマイズ。



**引数:**
 
 - `name`:  カスタマイズするメトリクス名。
 - `step_metric`:  このメトリクスの X 軸となる他のメトリクス名。 
 - `step_sync`:  `step_metric` が明示的に指定されていなければ自動で最後の値を挿入（デフォルトはTrue）。 
 - `hidden`:  このメトリクスを自動プロットで非表示にするか。 
 - `summary`:  サマリで追加する集計方法。"min"、"max"、"mean"、"last"、"best"、"copy"、"none" を指定可。"best" は goal パラメータと併用。"none"はサマリ非生成。"copy"は非推奨。 
 - `goal`:  "best" サマリ型をどう解釈するか。"minimize" または "maximize" を指定。 
 - `overwrite`:  False なら、同じメトリクスの既存 define_metric 呼び出しとマージし、省略した値は既存値が優先。True なら省略値も前回呼び出しで指定した値で上書き。



**戻り値:**
この呼び出しを表すオブジェクト（通常は不要でそのまま破棄可）。

---

### <kbd>method</kbd> `Run.display`

```python
display(height: 'int' = 420, hidden: 'bool' = False) → bool
```

この run を Jupyter 内で表示。

---

### <kbd>method</kbd> `Run.finish`

```python
finish(exit_code: 'int | None' = None, quiet: 'bool | None' = None) → None
```

run を完了し、残りのデータをアップロードします。

W&B run を完了状態にし、すべてのデータがサーバーに同期されることを保証します。run の最終状態は、終了条件と同期状況によって決まります。

Run ステータスの例: 
- Running: データを記録中／ハートビート送信中のアクティブな run
- Crashed: ハートビートが予期せず途絶えた run
- Finished: 正常終了（`exit_code=0`）し、すべて同期済み
- Failed: エラー終了（`exit_code!=0`）
- Killed: 強制終了され、完了できなかった run



**引数:**
 
 - `exit_code`:  run の終了ステータス（0なら成功、それ以外は失敗扱い）
 - `quiet`:  非推奨。ログ出力の詳細度は `wandb.Settings(quiet=...)` で設定してください。

---

### <kbd>method</kbd> `Run.finish_artifact`

```python
finish_artifact(
    artifact_or_path: 'Artifact | str',
    name: 'str | None' = None,
    type: 'str | None' = None,
    aliases: 'list[str] | None' = None,
    distributed_id: 'str | None' = None
) → Artifact
```

未確定のアーティファクトを run の出力として完了させます。

同じ distributed ID で以後 upsert すると新しいバージョンが作成されます。



**引数:**
 
 - `artifact_or_path`:  このアーティファクトの内容のパス。形式は: 
   - `/local/directory`
   - `/local/directory/file.txt`
   - `s3://bucket/path`
   または `wandb.Artifact` で作成した Artifact オブジェクト。
 - `name`:  アーティファクト名。entity/project でプレフィックス可。省略時はパス名+run id。
   - name:version
   - name:alias
   - digest
 - `type`:  アーティファクトのタイプ。例: `dataset`, `model`
 - `aliases`:  このアーティファクトに付与するエイリアス。デフォルトは `["latest"]`
 - `distributed_id`:  分散ジョブ共有の一意な文字列。None の場合は run のグループ名。



**戻り値:**
`Artifact` オブジェクト。

---




### <kbd>method</kbd> `Run.link_artifact`

```python
link_artifact(
    artifact: 'Artifact',
    target_path: 'str',
    aliases: 'list[str] | None' = None
) → Artifact | None
```

指定した artifact をポートフォリオ（アーティファクトのプロモートコレクション）にリンクします。

リンクされた artifact は指定したポートフォリオの UI で表示されます。



**引数:**
 
 - `artifact`:  リンクする artifact（パブリック or ローカル） 
 - `target_path`:  `str` 型。形式は `{portfolio}` 、`{project}/{portfolio}`、または `{entity}/{project}/{portfolio}`
 - `aliases`:  `List[str]` 型。ポートフォリオ内のこの artifact のみ有効なエイリアス（省略時は"latest"も自動付与）



**戻り値:**
リンクに成功した場合は artifact、失敗時は None

---

### <kbd>method</kbd> `Run.link_model`

```python
link_model(
    path: 'StrPath',
    registered_model_name: 'str',
    name: 'str | None' = None,
    aliases: 'list[str] | None' = None
) → Artifact | None
```

モデル artifact のバージョンを記録し、モデルレジストリ内の登録済みモデルにリンクします。

リンクされたモデルのバージョンは、指定された登録済みモデルの UI で確認できます。

このメソッドは以下を行います:
- `name` のモデル artifact が既に記録済みなら、そのファイルが `path` に一致する artifact バージョンを使う／無ければ新規記録。未設定なら、path のベース名＋run id で artifact 作成
- `registered_model_name` の registered model が 'model-registry' プロジェクトに無ければ、新規作成します
- モデル artifact 'name' のバージョンを 'registered_model_name' にリンク
- 'aliases'リスト内のエイリアスをこの artifact バージョンに付与



**引数:**
 
 - `path`:  (str) モデルデータのパス。以下いずれかの形式。
    - `/local/directory`
    - `/local/directory/file.txt`
    - `s3://bucket/path`
 - `registered_model_name`:  リンク先の登録済みモデル名。registered model は特定のMLタスクを表し、モデルバージョンのコレクションです。entity は run から自動取得されます。
 - `name`:  'path' のファイルを記録するモデル artifact の名前（省略時はデフォルト命名）。
 - `aliases`:  登録済みモデル内のリンク先 artifact に付与するエイリアス（latest は常に自動付与）



**例外:**
 
 - `AssertionError`:  registered_model_name にパスを指定した場合か、artifact 'name' に 'model' が含まれていない場合
 - `ValueError`:  name に無効な特殊文字



**戻り値:**
リンクに成功した場合は artifact、失敗時は `None`。

---

### <kbd>method</kbd> `Run.log`

```python
log(
    data: 'dict[str, Any]',
    step: 'int | None' = None,
    commit: 'bool | None' = None
) → None
```

run のデータをアップロードします。

`log` で run のデータ（スカラー、画像、動画、ヒストグラム、プロット、テーブル等）を記録します。[Log objects and media](https://docs.wandb.ai/guides/track/log) を参照すると、コード例やベストプラクティスが確認できます。

基本的な使い方:

```python
import wandb

with wandb.init() as run:
     run.log({"train-loss": 0.5, "accuracy": 0.9})
``` 

このコードスニペットは loss と accuracy を run の履歴へ記録し、これらのメトリクスの summary 値も更新します。

[wandb.ai](https://wandb.ai) の workspace、もしくは [セルフホスト](https://docs.wandb.ai/guides/hosting)した W&B アプリ、または [Public API](https://docs.wandb.ai/guides/track/public-api-guide) を活用して Jupyter ノートブックでも可視化・分析ができます。

記録する値はスカラーに限らず、[W&B の各種データ型](https://docs.wandb.ai/ref/python/data-types/)（画像、音声、動画など）を利用できます。構造化データは `wandb.Table` で記録できます。詳しくは [Log tables, visualize and query data](https://docs.wandb.ai/guides/models/tables/tables-walkthrough) を参照。

W&B では名前にスラッシュ（`/`）を含むメトリクスを、自動的に最後のスラッシュ前の文字列でセクション分けします。例:

```python
with wandb.init() as run:
     # "train" セクションと "validate" セクションで記録
     run.log(
         {
             "train/accuracy": 0.9,
             "train/loss": 30,
             "validate/accuracy": 0.8,
             "validate/loss": 20,
         }
     )
``` 

入れ子は1階層までサポート。`run.log({"a/b/c": 1})` だと "a/b" というセクションになります。

`run.log()` の呼び出し頻度は1秒あたり数回を超えないよう推奨します。パフォーマンスのため、N回に1度まとめて記録するか、何回かデータを収集して1ステップ分ずつログにまとめて記録しましょう。

デフォルトでは、`log` を呼ぶたび新たな「ステップ」が作成されます。step は単調増加する必要があり、過去の step への再記録はできません。グラフの X 軸には任意のメトリクスを指定可能。詳しくは [Custom log axes](https://docs.wandb.ai/guides/track/log/customize-logging-axes/) を参照。

多くの場合、W&B の step はトレーニング step というより「タイムスタンプ」として扱うのが便利です。

```python
with wandb.init() as run:
     # X 軸用にエポックの値を記録
     run.log({"epoch": 40, "train-loss": 0.5})
``` 

`step` と `commit` を使って、同じ step への複数回の log も可能。どの方法も同じ結果:

```python
with wandb.init() as run:
     # 通常
     run.log({"train-loss": 0.5, "accuracy": 0.8})
     run.log({"train-loss": 0.4, "accuracy": 0.9})

     # commit=False で自動カウントを止める
     run.log({"train-loss": 0.5}, commit=False)
     run.log({"accuracy": 0.8})
     run.log({"train-loss": 0.4}, commit=False)
     run.log({"accuracy": 0.9})

     # 明示的に step を管理
     run.log({"train-loss": 0.5}, step=current_step)
     run.log({"accuracy": 0.8}, step=current_step)
     current_step += 1
     run.log({"train-loss": 0.4}, step=current_step)
     run.log({"accuracy": 0.9}, step=current_step)
``` 



**引数:**
 
 - `data`:  `str` をキー、シリアライズ可能な Python オブジェクトを値に持つ `dict` 
 - `Python objects including`:  `int`、`float`、`string`、`wandb.data_types` のいずれか、またはシリアライズ可能なリスト/タプル/NumPy 配列、さらに同様の構造の `dict`。
 - `step`:  記録する step 番号。`None` なら自動カウントアップ。詳細は上記説明参照
 - `commit`:  True ならこの step 版を確定しアップロード。False なら step のデータを蓄積。詳しくは説明参照。`step` が None の時はデフォルト commit=True、それ以外は commit=False



**例:**
より詳細な例は [ガイド: ロギング](https://docs.wandb.com/guides/track/log) もご参照ください。

基本

```python
import wandb

with wandb.init() as run:
    run.log({"train-loss": 0.5, "accuracy": 0.9
``` 

増分記録

```python
import wandb

with wandb.init() as run:
    run.log({"loss": 0.2}, commit=False)
    # 別の所で step をまとめて記録
    run.log({"accuracy": 0.8})
``` 

ヒストグラム

```python
import numpy as np
import wandb

# 正規分布から勾配値をサンプリング
gradients = np.random.randn(100, 100)
with wandb.init() as run:
    run.log({"gradients": wandb.Histogram(gradients)})
``` 

NumPy 画像

```python
import numpy as np
import wandb

with wandb.init() as run:
    examples = []
    for i in range(3):
         pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
         image = wandb.Image(pixels, caption=f"random field {i}")
         examples.append(image)
    run.log({"examples": examples})
``` 

PIL 画像

```python
import numpy as np
from PIL import Image as PILImage
import wandb

with wandb.init() as run:
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

NumPy 動画

```python
import numpy as np
import wandb

with wandb.init() as run:
    # 軸: (time, channel, height, width)
    frames = np.random.randint(
         low=0,
         high=256,
         size=(10, 3, 100, 100),
         dtype=np.uint8,
    )
    run.log({"video": wandb.Video(frames, fps=4)})
``` 

Matplotlib グラフ

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

with wandb.init() as run:
    fig, ax = plt.subplots()
    x = np.linspace(0, 10)
    y = x * x
    ax.plot(x, y)  # y = x^2をプロット
    run.log({"chart": fig})
``` 

PR 曲線

```python
import wandb

with wandb.init() as run:
    run.log({"pr": wandb.plot.pr_curve(y_test, y_probas, labels)})
``` 

3D オブジェクト

```python
import wandb

with wandb.init() as run:
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



**例外:**
 
 - `wandb.Error`:  `wandb.init()` 前に呼び出した場合
 - `ValueError`:  無効なデータを渡した場合

---

### <kbd>method</kbd> `Run.log_artifact`

```python
log_artifact(
    artifact_or_path: 'Artifact | StrPath',
    name: 'str | None' = None,
    type: 'str | None' = None,
    aliases: 'list[str] | None' = None,
    tags: 'list[str] | None' = None
) → Artifact
```

アーティファクトを run の出力として宣言します。



**引数:**
 
 - `artifact_or_path`:  (str または Artifact) アーティファクト内容のパスまたは Artifact オブジェクト。例:
   - `/local/directory`
   - `/local/directory/file.txt`
   - `s3://bucket/path`
 - `name`:  (str, オプション) アーティファクト名。形式例:
   - name:version
   - name:alias
   - digest
   省略時はパス名＋run id。
 - `type`:  (str) 記録するアーティファクトのタイプ。例: `dataset`, `model`
 - `aliases`:  (list, オプション) このアーティファクトに適用するエイリアス（デフォルトは `["latest"]`）
 - `tags`:  (list, オプション) このアーティファクトに適用するタグ（オプション）



**戻り値:**
`Artifact` オブジェクト

---

### <kbd>method</kbd> `Run.log_code`

```python
log_code(
    root: 'str | None' = '.',
    name: 'str | None' = None,
    include_fn: 'Callable[[str, str], bool] | Callable[[str], bool]' = <function _is_py_requirements_or_dockerfile at 0x102da5f30>,
    exclude_fn: 'Callable[[str, str], bool] | Callable[[str], bool]' = <function exclude_wandb_fn at 0x103b4c5e0>
) → Artifact | None
```

現在のコードの状態を W&B Artifact として保存します。

デフォルトでは、カレントディレクトリー直下の `.py` ファイルを再帰的に全て記録します。



**引数:**
 
 - `root`:  コードを再帰的に検索する基準となる、`os.getcwd()` からの相対または絶対パス
 - `name`:  (str, オプション) コード artifact の名前。デフォルトは `source-$PROJECT_ID-$ENTRYPOINT_RELPATH`。複数の run で artifact 共有したい場合はここで指定してください。
 - `include_fn`:  ファイルパスと（オプションで）root を受け取り、記録する場合は True を返す callable。デフォルトは `lambda path, root: path.endswith(".py")`
 - `exclude_fn`:  ファイルパスと（オプションで）root を受け取り、除外する場合 True を返す callable。デフォルトで `<root>/.wandb/` や `<root>/wandb/` 配下を除外



**例:**
基本

```python
import wandb

with wandb.init() as run:
    run.log_code()
``` 

応用

```python
import wandb

with wandb.init() as run:
    run.log_code(
         root="../",
         include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
         exclude_fn=lambda path, root: os.path.relpath(path, root).startswith(
             "cache/"
         ),
    )
``` 



**戻り値:**
コード記録時は `Artifact` オブジェクト

---

### <kbd>method</kbd> `Run.log_model`

```python
log_model(
    path: 'StrPath',
    name: 'str | None' = None,
    aliases: 'list[str] | None' = None
) → None
```

モデル artifact を run に出力として記録し、`path` 配下の内容を含めます。

モデル artifact の名前には英数字/アンダースコア/ハイフンのみ使用可能です。



**引数:**
 
 - `path`:  (str) モデルデータへのパス。形式例:
    - `/local/directory`
    - `/local/directory/file.txt`
    - `s3://bucket/path`
 - `name`:  ファイル内容を追加する artifact 名（省略時はパス名＋run id）。
 - `aliases`:  作成された artifact に付与するエイリアス（デフォルトは `["latest"]`）



**例外:**
 
 - `ValueError`:  name に不正な特殊文字が含まれている場合



**戻り値:**
なし

---

### <kbd>method</kbd> `Run.mark_preempting`

```python
mark_preempting() → None
```

この run をプリエンプト中としてマークします。

内部プロセスにサーバーへ即時レポートを指示します。

---


### <kbd>method</kbd> `Run.restore`

```python
restore(
    name: 'str',
    run_path: 'str | None' = None,
    replace: 'bool' = False,
    root: 'str | None' = None
) → None | TextIO
```

指定したファイルをクラウドストレージからダウンロードします。

ファイルはカレントディレクトリーまたは run ディレクトリーに配置されます。デフォルトでは、ローカルにすでに存在する場合はダウンロードしません。



**引数:**
 
 - `name`:  ファイル名
 - `run_path`:  取得元 run へのパス（例: `username/project_name/run_id`）。`wandb.init` 未実行時は必須。
 - `replace`:  既存ファイルがあってもダウンロードするか
 - `root`:  ダウンロード先ディレクトリー（デフォルトはカレントまたは run ディレクトリー）



**戻り値:**
ファイルが見つからなければ None、それ以外は読み込み用のファイルオブジェクト



**例外:**
 
 - `CommError`:  W&B バックエンドへ接続できない場合
 - `ValueError`:  ファイル未発見もしくは run_path が見つからない場合

---

### <kbd>method</kbd> `Run.save`

```python
save(
    glob_str: 'str | os.PathLike',
    base_path: 'str | os.PathLike | None' = None,
    policy: 'PolicyName' = 'live'
) → bool | list[str]
```

1つまたは複数のファイルを W&B に同期します。

相対パスはカレントディレクトリー基準です。

Unix グロブ（"myfiles/*" など）は `save` 実行時に展開され、`policy` には依存しません。新規作成ファイルは自動検出されません。

`base_path` を指定するとファイルのディレクトリー構成を制御できます。`glob_str` のプレフィックスである必要があります。その下位ディレクトリー構造が保持されます。

絶対パスまたはグロブ＋未指定 `base_path` の場合、1階層分ディレクトリーが保存されます。

**引数:**
 
 - `glob_str`:  相対または絶対パス、または Unix グロブ
 - `base_path`:  ディレクトリー構造を決定するための基準パス（例は下記参照）
 - `policy`:  `live`、`now`、`end` のいずれか
    - live: ファイル変更時に随時アップロード（前バージョン上書き）
    - now: 今すぐファイル1回だけアップロード
    - end: run 終了時にアップロード



**戻り値:**
マッチしたファイルのシンボリックリンクへのパス

過去の理由から、従来コードでは bool を返す場合もあります

```python
import wandb

run = wandb.init()

run.save("these/are/myfiles/*")
# => run 内で "these/are/myfiles/" フォルダーに保存

run.save("these/are/myfiles/*", base_path="these")
# => run 内で "are/myfiles/" フォルダーに保存

run.save("/User/username/Documents/run123/*.txt")
# => run 内で "run123/" フォルダーに保存（詳細は下記参照）

run.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => run 内で "username/Documents/run123/" フォルダーに保存

run.save("files/*/saveme.txt")
# => "files/" の各サブディレクトリー内に "saveme.txt" を適したサブディレクトリに保存

# コンテキストマネージャを使用していない場合は明示的に run を終了
run.finish()
``` 

---

### <kbd>method</kbd> `Run.status`

```python
status() → RunStatus
```

この run の同期ステータスなど内部バックエンドから同期情報を取得します。

---


### <kbd>method</kbd> `Run.unwatch`

```python
unwatch(
    models: 'torch.nn.Module | Sequence[torch.nn.Module] | None' = None
) → None
```

pytorch モデルのトポロジー・勾配・パラメータのフックを全て解除します。



**引数:**
 
 - `models`:  watch した pytorch モデル（リスト可）

---

### <kbd>method</kbd> `Run.upsert_artifact`

```python
upsert_artifact(
    artifact_or_path: 'Artifact | str',
    name: 'str | None' = None,
    type: 'str | None' = None,
    aliases: 'list[str] | None' = None,
    distributed_id: 'str | None' = None
) → Artifact
```

未確定のアーティファクトを run の出力として宣言（または追記）します。

finalize には run.finish_artifact() の呼び出しが必須です。分散ジョブなどで全てが同じ artifact に貢献する場合に有用です。



**引数:**
 
 - `artifact_or_path`:  アーティファクト内容のパス。例:
    - `/local/directory`
    - `/local/directory/file.txt`
    - `s3://bucket/path`
 - `name`:  アーティファクト名。"entity/project" でプレフィックス可。省略時はパス名＋run ID。形式例は:
    - name:version
    - name:alias
    - digest
 - `type`:  アーティファクトのタイプ。例: `dataset`, `model`
 - `aliases`:  このアーティファクトに付与するエイリアス（デフォルトは `["latest"]`）
 - `distributed_id`:  分散ジョブ共有の一意な文字列。None の場合は run のグループ名



**戻り値:**
`Artifact` オブジェクト

---

### <kbd>method</kbd> `Run.use_artifact`

```python
use_artifact(
    artifact_or_name: 'str | Artifact',
    type: 'str | None' = None,
    aliases: 'list[str] | None' = None,
    use_as: 'str | None' = None
) → Artifact
```

アーティファクトを run の入力として宣言します。

返されたオブジェクトから `download` や `file` を呼んでファイル内容をローカルに取得できます。



**引数:**
 
 - `artifact_or_name`:  使用したいアーティファクト名。プロジェクト名（`<entity>` または `<entity>/<project>`）でプレフィックス可能。entity を指定しない場合、run/API設定の entity が使用されます。形式例:
    - name:version
    - name:alias
 - `type`:  使用するアーティファクトのタイプ
 - `aliases`:  このアーティファクトに付与するエイリアス
 - `use_as`:  非推奨。動作しません



**戻り値:**
`Artifact` オブジェクト



**例:**
 ```python
import wandb

run = wandb.init(project="<example>")

# エイリアスで artifact を利用
artifact_a = run.use_artifact(artifact_or_name="<name>:<alias>")

# バージョン指定で利用
artifact_b = run.use_artifact(artifact_or_name="<name>:v<version>")

# entity, project, name, alias 指定で利用
artifact_c = run.use_artifact(
    artifact_or_name="<entity>/<project>/<name>:<alias>"
)

# entity, project, name, version 指定で利用
artifact_d = run.use_artifact(
    artifact_or_name="<entity>/<project>/<name>:v<version>"
)

# コンテキストマネージャを使わない場合は finish() 必須
run.finish()
``` 

---

### <kbd>method</kbd> `Run.use_model`

```python
use_model(name: 'str') → FilePathStr
```

`name` で指定したモデル artifact のファイル群をダウンロードします。



**引数:**
 
 - `name`:  モデル artifact 名。既に記録済み artifact 名と一致が必要。`entity/project/` でのプレフィックスも可。形式例:
    - model_artifact_name:version
    - model_artifact_name:alias



**戻り値:**
 
 - `path` (str):  ダウンロードしたモデル artifact のファイルへのパス



**例外:**
 
 - `AssertionError`:  'name' artifact のタイプに 'model' が含まれていない場合

---

### <kbd>method</kbd> `Run.watch`

```python
watch(
    models: 'torch.nn.Module | Sequence[torch.nn.Module]',
    criterion: 'torch.F | None' = None,
    log: "Literal['gradients', 'parameters', 'all'] | None" = 'gradients',
    log_freq: 'int' = 1000,
    idx: 'int | None' = None,
    log_graph: 'bool' = False
) → None
```

指定した PyTorch モデルにフックし、勾配や計算グラフを監視します。

パラメータや勾配のどちらか、または両方をトレーニング時に記録できます。



**引数:**
 
 - `models`:  監視対象のモデル（単体またはリスト）
 - `criterion`:  最適化に使う損失関数（オプション）
 - `log`:  記録対象を "gradients"、"parameters"、"all" から選択。Noneは記録オフ（デフォルトは"gradients"）
 - `log_freq`:  勾配・パラメータを記録するバッチ数間隔（デフォルト1000）
 - `idx`:  `wandb.watch` で複数モデルを監視する場合のインデックス（デフォルトNone）
 - `log_graph`:  モデルの計算グラフも記録するか（デフォルトFalse）



**例外:**
ValueError:  `wandb.init()` が未呼び出しか、指定モデルが `torch.nn.Module` インスタンスでない場合