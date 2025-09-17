---
title: Run
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-classes-Run
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py >}}




## <kbd>class</kbd> `Run`
W&B によってログされる計算の単位。通常は ML の experiment です。

新しい run を作成するには [`wandb.init()`](https://docs.wandb.ai/ref/python/init/) を呼び出します。`wandb.init()` は新しい run を開始して `wandb.Run` オブジェクトを返します。各 run には一意の ID（run ID）が割り当てられます。run の終了を自動化するために、コンテキスト（`with` 文）マネージャーを使うことを W&B は推奨します。

分散トレーニング experiment の場合、各プロセスにつき 1 つの run で別々にトラッキングするか、すべてのプロセスを 1 つの run にまとめてトラッキングできます。詳細は [Log distributed training experiments](https://docs.wandb.ai/guides/track/log/distributed-training) を参照してください。

`wandb.Run.log()` で run にデータをログできます。`wandb.Run.log()` でログしたものは、その run に送信されます。詳しくは [Create an experiment](https://docs.wandb.ai/guides/track/create-an-experiment/) または [`wandb.init`](https://docs.wandb.ai/ref/python/init/) の API リファレンスを参照してください。

[`wandb.apis.public`](https://docs.wandb.ai/ref/python/public-api/api/) 名前空間にも別の `Run` オブジェクトがあります。これは、すでに作成済みの run とやり取りするために使用します。



**属性:**
 
 - `summary`:  （Summary）run のサマリー。辞書のように扱えるオブジェクトです。詳細は 
 - `[Log summary metrics](https`: //docs.wandb.ai/guides/track/log/log-summary/)。 



**例:**
 `wandb.init()` で run を作成: 

```python
import wandb

# 新しい run を開始してデータをログする
# コンテキストマネージャー（`with` 文）で run を自動終了
with wandb.init(entity="entity", project="project") as run:
    run.log({"accuracy": acc, "loss": loss})
``` 


### <kbd>property</kbd> Run.config

この run に関連付けられた Config オブジェクト。 

---

### <kbd>property</kbd> Run.config_static

この run に関連付けられた静的 Config オブジェクト。 

---

### <kbd>property</kbd> Run.dir

run に関連するファイルが保存されるディレクトリー。 

---

### <kbd>property</kbd> Run.disabled

run が無効なら True、そうでなければ False。 

---

### <kbd>property</kbd> Run.entity

この run に関連付けられた W&B の Entity 名。 

Entity はユーザー名、チーム名、あるいは組織名になり得ます。 

---

### <kbd>property</kbd> Run.group

この run に関連付けられたグループ名を返します。 

run をグループ化すると、関連する experiment を W&B UI でまとめて整理・可視化できます。特に、分散トレーニングやクロスバリデーションのように、複数の run を 1 つの experiment として扱いたい場面で有用です。 

共有モード（すべてのプロセスで同じ run オブジェクトを共有）では、run が 1 つだけでグループ化が不要なため、通常は group を設定する必要はありません。 

---

### <kbd>property</kbd> Run.id

この run の識別子。 

---

### <kbd>property</kbd> Run.job_type

run に関連付けられたジョブタイプ名。 

ジョブタイプは W&B App の run の Overview ページで確認できます。 

ジョブタイプ（例: "training"、"evaluation"、"inference" など）で run をカテゴリ分けできます。同一 Project 内で異なるジョブタイプの run が多数ある場合などに、W&B UI での整理やフィルタリングに便利です。詳細は [Organize runs](https://docs.wandb.ai/guides/runs/#organize-runs) を参照してください。 

---

### <kbd>property</kbd> Run.name

run の表示名。 

表示名は一意である保証はなく、説明的な名称にできます。デフォルトではランダムに生成されます。 

---

### <kbd>property</kbd> Run.notes

run に紐づくノート（あれば）。 

ノートは複数行の文字列にでき、`$$` の中で markdown や latex 方程式も使用できます（例: `$x + 3$`）。 

---

### <kbd>property</kbd> Run.offline

run がオフラインなら True、そうでなければ False。 

---

### <kbd>property</kbd> Run.path

run へのパス。 

run のパスには Entity、Project、run ID が含まれ、形式は `entity/project/run_id` です。 

---

### <kbd>property</kbd> Run.project

この run に関連付けられた W&B Project 名。 

---

### <kbd>property</kbd> Run.project_url

この run に関連付けられた W&B Project の URL（あれば）。 

オフラインの run には Project URL はありません。 

---

### <kbd>property</kbd> Run.resumed

run が再開された場合は True、そうでなければ False。 

---

### <kbd>property</kbd> Run.settings

run の Settings オブジェクトの凍結コピー。 

---

### <kbd>property</kbd> Run.start_time

run 開始時刻の Unix タイムスタンプ（秒）。 

---



### <kbd>property</kbd> Run.sweep_id

この run に関連付けられた sweep の識別子（あれば）。 

---

### <kbd>property</kbd> Run.sweep_url

この run に関連付けられた sweep の URL（あれば）。 

オフラインの run には sweep URL はありません。 

---

### <kbd>property</kbd> Run.tags

run に関連付けられたタグ（あれば）。 

---

### <kbd>property</kbd> Run.url

W&B の run の URL（あれば）。 

オフラインの run には URL はありません。 



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
 
 - `title`:  アラートのタイトル。64 文字未満である必要があります。 
 - `text`:  アラートの本文。 
 - `level`:  使用するアラートレベル。`INFO`、`WARN`、`ERROR` のいずれか。 
 - `wait_duration`:  同じタイトルのアラートを再送するまでに待つ時間（秒）。 

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

`wandb.Run.log()` でログされるメトリクスをカスタマイズします。 



**引数:**
 
 - `name`:  カスタマイズするメトリクス名。 
 - `step_metric`:  自動生成チャートでこのメトリクスの X 軸として使う別のメトリクス名。 
 - `step_sync`:  `wandb.Run.log()` に `step_metric` の最新値が明示されていない場合、自動で挿入します。`step_metric` が指定されていればデフォルトは True。 
 - `hidden`:  このメトリクスを自動プロットから非表示にします。 
 - `summary`:  summary に追加する集約メトリクスを指定します。サポートされる集約は "min"、"max"、"mean"、"last"、"first"、"best"、"copy"、"none"。"none" は summary の生成を抑止します。"best" は `goal` パラメータと併用しますが、非推奨なので "min" か "max" を使用してください。"copy" も非推奨です。 
 - `goal`:  "best" summary の解釈方法を指定します。"minimize" と "maximize" をサポートしますが、`goal` は非推奨なので "min" か "max" を使用してください。 
 - `overwrite`:  false の場合、同じメトリクスに対する過去の `define_metric` 呼び出しとマージし、未指定パラメータには過去の値を使用します。true の場合、未指定パラメータは過去の値を上書きします。 



**戻り値:**
 この呼び出しを表すオブジェクト。通常は破棄して構いません。 

---

### <kbd>method</kbd> `Run.display`

```python
display(height: 'int' = 420, hidden: 'bool' = False) → bool
```

この run を Jupyter で表示します。 

---

### <kbd>method</kbd> `Run.finish`

```python
finish(exit_code: 'int | None' = None, quiet: 'bool | None' = None) → None
```

run を終了し、残りのデータをアップロードします。 

W&B の run の完了をマークし、すべてのデータがサーバーに同期されることを保証します。run の最終状態は終了条件と同期状況で決まります。 

Run の状態: 
- Running: データのログやハートビート送信を行っている実行中の run。 
- Crashed: ハートビートの送信が予期せず停止した run。 
- Finished: `exit_code=0` で正常終了し、すべてのデータが同期済みの run。 
- Failed: `exit_code!=0` でエラー終了した run。 
- Killed: 完了前に強制停止された run。 



**引数:**
 
 - `exit_code`:  終了ステータス。成功は 0、その他は失敗としてマークされます。 
 - `quiet`:  非推奨。ログの冗長性は `wandb.Settings(quiet=...)` で設定してください。 

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

未確定の Artifact を run の出力として確定します。 

同じ distributed ID で以降に「upsert」すると、新しいバージョンが作成されます。 



**引数:**
 
 - `artifact_or_path`:  この Artifact の内容へのパス。以下の形式が使えます: 
            - `/local/directory` 
            - `/local/directory/file.txt` 
            - `s3://bucket/path`  また、`wandb.Artifact` を呼び出して作成した Artifact オブジェクトも渡せます。 
 - `name`:  Artifact 名。先頭に `entity/project` を付けても構いません。有効な名前は次の形式です: 
            - name:version 
            - name:alias 
            - digest  省略時は、パスのベース名の前に現在の run id を付けたものがデフォルトになります。 
 - `type`:  ログする Artifact のタイプ。例: `dataset`、`model` 
 - `aliases`:  この Artifact に適用するエイリアス。デフォルトは `["latest"]`。 
 - `distributed_id`:  すべての分散ジョブが共有する一意の文字列。None の場合、run の group 名がデフォルトになります。 



**戻り値:**
 `Artifact` オブジェクト。 

---




### <kbd>method</kbd> `Run.link_artifact`

```python
link_artifact(
    artifact: 'Artifact',
    target_path: 'str',
    aliases: 'list[str] | None' = None
) → Artifact
```

指定した Artifact をポートフォリオ（昇格済み Artifact のコレクション）にリンクします。 

リンクされた Artifact は、指定したポートフォリオの UI に表示されます。 



**引数:**
 
 - `artifact`:  リンクする（公開またはローカルの）Artifact。 
 - `target_path`:  `str` - 次の形式を取ります: `{portfolio}`、`{project}/{portfolio}`、`{entity}/{project}/{portfolio}` 
 - `aliases`:  `List[str]` - 任意。ポートフォリオ内のこのリンク済み Artifact にのみ適用されるエイリアス。エイリアス "latest" はリンクされた Artifact の最新バージョンに常に適用されます。 



**戻り値:**
 リンクされた Artifact。 

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

モデル Artifact のバージョンをログし、Model Registry の Registered Model にリンクします。 

リンクされたモデルバージョンは、指定した Registered Model の UI に表示されます。 

このメソッドは以下を行います: 
- 'name' のモデル Artifact がすでにログされているか確認します。ログ済みなら、'path' のファイルと一致するバージョンを使用するか、新しいバージョンをログします。未ログの場合は 'path' 以下のファイルをタイプ 'model' のモデル Artifact 'name' として新規ログします。 
- 'model-registry' Project 内に 'registered_model_name' の Registered Model が存在するか確認します。なければ 'registered_model_name' の Registered Model を新規作成します。 
- モデル Artifact 'name' のバージョンを Registered Model 'registered_model_name' にリンクします。 
- 'aliases' のリストにあるエイリアスを、新たにリンクされたモデル Artifact バージョンに付与します。 



**引数:**
 
 - `path`:  （str）このモデルの内容へのパス。以下の形式が使えます: 
    - `/local/directory` 
    - `/local/directory/file.txt` 
    - `s3://bucket/path` 
 - `registered_model_name`:  このモデルをリンクする Registered Model の名前。Registered Model は Model Registry にリンクされたモデルバージョンの集合で、通常はチームの特定の ML タスクを表します。この Registered Model が属する Entity は run から推測されます。 
 - `name`:  'path' 内のファイルをログするモデル Artifact 名。省略時は、パスのベース名の前に現在の run id が付きます。 
 - `aliases`:  Registered Model 内のこのリンク済み Artifact にのみ適用されるエイリアス。エイリアス "latest" はリンクされた Artifact の最新バージョンに常に適用されます。 



**例外:**
 
 - `AssertionError`:  `registered_model_name` がパスである場合、またはモデル Artifact 'name' のタイプに 'model' が含まれない場合。 
 - `ValueError`:  name に不正な特殊文字が含まれる場合。 



**戻り値:**
 リンクに成功した場合はリンクされた Artifact。失敗した場合は `None`。 

---

### <kbd>method</kbd> `Run.log`

```python
log(
    data: 'dict[str, Any]',
    step: 'int | None' = None,
    commit: 'bool | None' = None
) → None
```

run データをアップロードします。 

`log` を使って、スカラー、画像、動画、ヒストグラム、プロット、テーブルなどのデータを run からログします。コードスニペットやベストプラクティスなどは [Log objects and media](https://docs.wandb.ai/guides/track/log) を参照してください。 

基本的な使い方: 

```python
import wandb

with wandb.init() as run:
     run.log({"train-loss": 0.5, "accuracy": 0.9})
``` 

上記のコードスニペットは、loss と accuracy を run の履歴に保存し、これらのメトリクスの summary 値を更新します。 

ログしたデータは [wandb.ai](https://wandb.ai) の Workspace、または W&B アプリの [self-hosted instance](https://docs.wandb.ai/guides/hosting) で可視化できます。あるいはデータをエクスポートしてローカル（Jupyter ノートブックなど）で可視化・探索することもでき、その場合は [Public API](https://docs.wandb.ai/guides/track/public-api-guide) を利用します。 

ログする値はスカラーに限りません。画像、音声、動画など、任意の [W&B supported Data Type](https://docs.wandb.ai/ref/python/data-types/) をログできます。たとえば `wandb.Table` を使って構造化データをログできます。詳細はチュートリアル [Log tables, visualize and query data](https://docs.wandb.ai/guides/models/tables/tables-walkthrough) を参照してください。 

W&B は、名前にスラッシュ（`/`）を含むメトリクスを、最後のスラッシュより前の文字列を使ってセクションに整理します。たとえば、次の例では "train" と "validate" という 2 つのセクションに分かれます: 

```python
with wandb.init() as run:
     # "train" セクションにメトリクスをログ
     run.log(
         {
             "train/accuracy": 0.9,
             "train/loss": 30,
             "validate/accuracy": 0.8,
             "validate/loss": 20,
         }
     )
``` 

ネストは 1 階層のみサポートされます。`run.log({"a/b/c": 1})` は "a/b" というセクションになります。 

`run.log()` は 1 秒間に何度も呼ぶことを意図していません。最適な性能のため、N イテレーションごとに 1 回のログにするか、複数イテレーション分のデータを集めて 1 ステップでまとめてログしてください。 

デフォルトでは、`log` を呼ぶたびに新しい「step」が作成されます。step は常に増加しなければならず、過去の step に対してログすることはできません。任意のメトリクスをチャートの X 軸として使用できます。詳細は [Custom log axes](https://docs.wandb.ai/guides/track/log/customize-logging-axes/) を参照してください。 

多くの場合、W&B の step はトレーニングステップというよりもタイムスタンプのように扱う方がよいでしょう。 

```python
with wandb.init() as run:
     # 例: X 軸として使う "epoch" メトリクスをログ
     run.log({"epoch": 40, "train-loss": 0.5})
``` 

`step` と `commit` パラメータを使うと、複数回の `wandb.Run.log()` 呼び出しで同じ step にログできます。以下はいずれも同等です: 

```python
with wandb.init() as run:
     # 通常の使い方:
     run.log({"train-loss": 0.5, "accuracy": 0.8})
     run.log({"train-loss": 0.4, "accuracy": 0.9})

     # 暗黙の step（自動インクリメントなし）:
     run.log({"train-loss": 0.5}, commit=False)
     run.log({"accuracy": 0.8})
     run.log({"train-loss": 0.4}, commit=False)
     run.log({"accuracy": 0.9})

     # 明示的な step:
     run.log({"train-loss": 0.5}, step=current_step)
     run.log({"accuracy": 0.8}, step=current_step)
     current_step += 1
     run.log({"train-loss": 0.4}, step=current_step)
     run.log({"accuracy": 0.9}, step=current_step)
``` 



**引数:**
 
 - `data`:  キーが `str`、値がシリアライズ可能な Python オブジェクトの `dict` 
 - `Python objects including`:  `int`、`float`、`string`、`wandb.data_types` のいずれか、シリアライズ可能な Python オブジェクトのリスト、タプル、NumPy 配列、同様の構造の `dict`。 
 - `step`:  ログする step 番号。`None` の場合は暗黙の自動インクリメント step が使われます。詳細は説明を参照。 
 - `commit`:  true ならこの step を確定してアップロードします。false ならこの step のデータを貯めます。詳細は説明を参照。`step` が `None` の場合のデフォルトは `commit=True`、それ以外は `commit=False`。 



**例:**
 より多くの詳細な例は [our guides to logging](https://docs.wandb.com/guides/track/log) を参照してください。 

基本的な使い方 

```python
import wandb

with wandb.init() as run:
    run.log({"train-loss": 0.5, "accuracy": 0.9
``` 

インクリメンタルなロギング 

```python
import wandb

with wandb.init() as run:
    run.log({"loss": 0.2}, commit=False)
    # 準備ができたらこの step を確定して報告
    run.log({"accuracy": 0.8})
``` 

ヒストグラム 

```python
import numpy as np
import wandb

# 正規分布に従う乱数から勾配をサンプリング
gradients = np.random.randn(100, 100)
with wandb.init() as run:
    run.log({"gradients": wandb.Histogram(gradients)})
``` 

NumPy から画像 

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

PIL から画像 

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

NumPy から動画 

```python
import numpy as np
import wandb

with wandb.init() as run:
    # 軸は (time, channel, height, width)
    frames = np.random.randint(
         low=0,
         high=256,
         size=(10, 3, 100, 100),
         dtype=np.uint8,
    )
    run.log({"video": wandb.Video(frames, fps=4)})
``` 

Matplotlib のプロット 

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

with wandb.init() as run:
    fig, ax = plt.subplots()
    x = np.linspace(0, 10)
    y = x * x
    ax.plot(x, y)  # y = x^2 をプロット
    run.log({"chart": fig})
``` 

PR カーブ 

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
 
 - `wandb.Error`:  `wandb.init()` より前に呼び出された場合。 
 - `ValueError`:  不正なデータが渡された場合。 

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

run の出力として Artifact を宣言します。 



**引数:**
 
 - `artifact_or_path`:  （str または Artifact）この Artifact の内容へのパス。以下の形式が使えます: 
            - `/local/directory` 
            - `/local/directory/file.txt` 
            - `s3://bucket/path`  また、`wandb.Artifact` を呼び出して作成した Artifact オブジェクトも渡せます。 
 - `name`:  （任意の str）Artifact 名。有効な名前は次の形式です: 
            - name:version 
            - name:alias 
            - digest  省略時は、パスのベース名の前に現在の run id を付けたものがデフォルトになります。 
 - `type`:  （str）ログする Artifact のタイプ。例: `dataset`、`model` 
 - `aliases`:  （任意の list）この Artifact に適用するエイリアス。デフォルトは `["latest"]`。 
 - `tags`:  （任意の list）この Artifact に適用するタグ（任意）。 



**戻り値:**
 `Artifact` オブジェクト。 

---

### <kbd>method</kbd> `Run.log_code`

```python
log_code(
    root: 'str | None' = '.',
    name: 'str | None' = None,
    include_fn: 'Callable[[str, str], bool] | Callable[[str], bool]' = <function _is_py_requirements_or_dockerfile at 0x10342a8c0>,
    exclude_fn: 'Callable[[str, str], bool] | Callable[[str], bool]' = <function exclude_wandb_fn at 0x1050f4ee0>
) → Artifact | None
```

現在のコード状態を W&B の Artifact に保存します。 

デフォルトでは、カレントディレクトリーを走査し、`.py` で終わるすべてのファイルをログします。 



**引数:**
 
 - `root`:  コードを再帰的に探索するための `os.getcwd()` からの相対パス、または絶対パス。 
 - `name`:  （任意の str）コード Artifact の名前。デフォルトでは `source-$PROJECT_ID-$ENTRYPOINT_RELPATH` という名前になります。複数の run で同じ Artifact を共有したい場合など、`name` を指定できます。 
 - `include_fn`:  ファイルパスと（任意で）ルートパスを受け取り、含める場合は True、そうでなければ False を返す呼び出し可能オブジェクト。 
 - `defaults to `lambda path, root`:  path.endswith(".py")`. 
 - `exclude_fn`:  ファイルパスと（任意で）ルートパスを受け取り、除外する場合は `True`、そうでなければ `False` を返す呼び出し可能オブジェクト。デフォルトでは `<root>/.wandb/` および `<root>/wandb/` ディレクトリー内のすべてのファイルを除外します。 



**例:**
 基本的な使い方 

```python
import wandb

with wandb.init() as run:
    run.log_code()
``` 

高度な使い方 

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
 コードがログされた場合は `Artifact` オブジェクト。 

---

### <kbd>method</kbd> `Run.log_model`

```python
log_model(
    path: 'StrPath',
    name: 'str | None' = None,
    aliases: 'list[str] | None' = None
) → None
```

'path' 内の内容を含むモデル Artifact を run にログし、この run の出力としてマークします。 

モデル Artifact の名前には英数字、アンダースコア、ハイフンのみを使用できます。 



**引数:**
 
 - `path`:  （str）このモデルの内容へのパス。以下の形式が使えます: 
            - `/local/directory` 
            - `/local/directory/file.txt` 
            - `s3://bucket/path` 
 - `name`:  ファイル内容を追加するモデル Artifact の名前。省略時は、パスのベース名の前に現在の run id が付きます。 
 - `aliases`:  作成されるモデル Artifact に適用するエイリアス。デフォルトは `["latest"]`。 



**例外:**
 
 - `ValueError`:  name に不正な特殊文字が含まれる場合。 



**戻り値:**
 なし 

---

### <kbd>method</kbd> `Run.mark_preempting`

```python
mark_preempting() → None
```

この run を preempting としてマークします。 

内部プロセスにも、これをサーバーへ即時報告するよう指示します。 

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

クラウドストレージから指定したファイルをダウンロードします。 

ファイルはカレントディレクトリーまたは run ディレクトリーに配置されます。デフォルトでは、ローカルに存在しない場合のみダウンロードします。 



**引数:**
 
 - `name`:  ファイル名。 
 - `run_path`:  ファイルを取得する run のパス（例: `username/project_name/run_id`）。`wandb.init` が未呼び出しの場合は必須。 
 - `replace`:  ローカルに同名ファイルが存在する場合でもダウンロードするかどうか。 
 - `root`:  ファイルをダウンロードするディレクトリー。デフォルトはカレントディレクトリー（`wandb.init` を呼んだ場合は run ディレクトリー）。 



**戻り値:**
 ファイルが見つからない場合は None、見つかった場合は読み取り用に開いたファイルオブジェクト。 



**例外:**
 
 - `CommError`:  W&B がバックエンドへ接続できない場合。 
 - `ValueError`:  ファイルが見つからない、または `run_path` が見つからない場合。 

---

### <kbd>method</kbd> `Run.save`

```python
save(
    glob_str: 'str | os.PathLike',
    base_path: 'str | os.PathLike | None' = None,
    policy: 'PolicyName' = 'live'
) → bool | list[str]
```

1 つ以上のファイルを W&B に同期します。 

相対パスはカレントワーキングディレクトリーからの相対です。 

"myfiles/*" のような Unix グロブは、`policy` に関係なく `save` を呼び出した時点で展開されます。特に、新しいファイルは自動的には拾われません。 

アップロードするファイルのディレクトリー構造を制御するために `base_path` を指定できます。これは `glob_str` の接頭辞である必要があり、その下のディレクトリー構造は保持されます。 

絶対パスまたは絶対グロブを `base_path` なしで渡すと、上位 1 階層のディレクトリーは（下記の例のように）保持されます。 



**引数:**
 
 - `glob_str`:  相対または絶対パス、または Unix グロブ。 
 - `base_path`:  ディレクトリー構造の推定に使うパス。例を参照。 
 - `policy`:  `live`、`now`、`end` のいずれか。 
    - live: 変更のたびにアップロードし、以前のバージョンを上書き 
    - now: その場で 1 回だけアップロード 
    - end: run の終了時にアップロード 



**戻り値:**
 マッチしたファイルに対して作成されたシンボリックリンクのパス。 

過去との互換性のため、レガシーコードでは boolean を返すことがあります。 

```python
import wandb

run = wandb.init()

run.save("these/are/myfiles/*")
# => run の中の "these/are/myfiles/" フォルダーに保存されます。

run.save("these/are/myfiles/*", base_path="these")
# => run の中の "are/myfiles/" フォルダーに保存されます。

run.save("/User/username/Documents/run123/*.txt")
# => run の中の "run123/" フォルダーに保存されます。下記の注意を参照。

run.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => run の中の "username/Documents/run123/" フォルダーに保存されます。

run.save("files/*/saveme.txt")
# => それぞれの "saveme.txt" は "files/" の適切なサブディレクトリーに保存されます。

# コンテキストマネージャーを使っていないので明示的に run を終了
run.finish()
``` 

---

### <kbd>method</kbd> `Run.status`

```python
status() → RunStatus
```

内部バックエンドから、この run の同期ステータスに関する情報を取得します。 

---


### <kbd>method</kbd> `Run.unwatch`

```python
unwatch(
    models: 'torch.nn.Module | Sequence[torch.nn.Module] | None' = None
) → None
```

PyTorch のモデルのトポロジー、勾配、パラメータのフックを削除します。 



**引数:**
 
 - `models`:  `watch` を適用済みの PyTorch モデルの任意リスト。 

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

未確定の Artifact を run の出力として宣言（または追記）します。 

Artifact を確定するには `run.finish_artifact()` を呼ぶ必要があります。これは、分散ジョブが同じ Artifact に寄与する必要がある場合に便利です。 



**引数:**
 
 - `artifact_or_path`:  この Artifact の内容へのパス。以下の形式が使えます: 
    - `/local/directory` 
    - `/local/directory/file.txt` 
    - `s3://bucket/path` 
 - `name`:  Artifact 名。先頭に `"entity/project"` を付けても構いません。省略時は、パスのベース名の前に現在の run ID を付けたものがデフォルト。有効な名前は次の形式です: 
    - name:version 
    - name:alias 
    - digest 
 - `type`:  ログする Artifact のタイプ。一般的な例: `dataset`、`model`。 
 - `aliases`:  この Artifact に適用するエイリアス。デフォルトは `["latest"]`。 
 - `distributed_id`:  すべての分散ジョブが共有する一意の文字列。None の場合、run の group 名がデフォルトになります。 



**戻り値:**
 `Artifact` オブジェクト。 

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

run の入力として Artifact を宣言します。 

戻り値のオブジェクトに対して `download` または `file` を呼ぶと、内容をローカルに取得できます。 



**引数:**
 
 - `artifact_or_name`:  使用する Artifact の名前。Artifact がログされた Project 名で接頭することができます（"<entity>" または "<entity>/<project>"）。名前に Entity を指定しない場合、Run または API の設定で指定された Entity が使われます。有効な名前の形式: 
    - name:version 
    - name:alias 
 - `type`:  使用する Artifact のタイプ。 
 - `aliases`:  この Artifact に適用するエイリアス。 
 - `use_as`:  この引数は非推奨で、何もしません。 



**戻り値:**
 `Artifact` オブジェクト。 



**例:**
 ```python
import wandb

run = wandb.init(project="<example>")

# エイリアス指定で Artifact を使用
artifact_a = run.use_artifact(artifact_or_name="<name>:<alias>")

# バージョン指定で Artifact を使用
artifact_b = run.use_artifact(artifact_or_name="<name>:v<version>")

# entity/project/name:alias で Artifact を使用
artifact_c = run.use_artifact(
    artifact_or_name="<entity>/<project>/<name>:<alias>"
)

# entity/project/name:version で Artifact を使用
artifact_d = run.use_artifact(
    artifact_or_name="<entity>/<project>/<name>:v<version>"
)

# コンテキストマネージャーを使っていないので明示的に run を終了
run.finish()
``` 

---

### <kbd>method</kbd> `Run.use_model`

```python
use_model(name: 'str') → FilePathStr
```

モデル Artifact 'name' にログされたファイルをダウンロードします。 



**引数:**
 
 - `name`:  モデル Artifact 名。'name' は既存のモデル Artifact 名と一致している必要があります。先頭に `entity/project/` を付けても構いません。有効な名前の形式: 
    - model_artifact_name:version 
    - model_artifact_name:alias 



**戻り値:**
 
 - `path`（str）:  ダウンロードされたモデル Artifact のファイルへのパス。 



**例外:**
 
 - `AssertionError`:  モデル Artifact 'name' のタイプに 'model' が含まれない場合。 

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

指定した PyTorch モデルにフックして、勾配とモデルの計算グラフを監視します。 

この関数は、トレーニング中のパラメータ、勾配、または両方をトラッキングできます。 



**引数:**
 
 - `models`:  監視対象の単一モデル、または複数モデルのシーケンス。 
 - `criterion`:  最適化対象の損失関数（任意）。 
 - `log`:  "gradients"、"parameters"、"all" のいずれをログするか。無効化するには None を指定。（デフォルトは "gradients"） 
 - `log_freq`:  勾配とパラメータをログする頻度（バッチ数）。（デフォルト: 1000） 
 - `idx`:  複数モデルを `wandb.watch` でトラッキングする際のインデックス。（デフォルト: None） 
 - `log_graph`:  モデルの計算グラフをログするかどうか。（デフォルト: False） 



**例外:**
 ValueError:  `wandb.init()` が呼ばれていない場合、またはモデルが `torch.nn.Module` のインスタンスでない場合。