---
title: run
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-classes-Run
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py >}}




## <kbd>class</kbd> `Run`
W&B で記録される計算単位です。一般的には ML の実験を指します。

新しい run を作成するには [`wandb.init()`](https://docs.wandb.ai/ref/python/init/) を呼び出します。`wandb.init()` を実行すると run がスタートし、`wandb.Run` オブジェクトが返されます。それぞれの run には一意な ID（run ID）が割り振られます。W&B ではコンテキスト管理（`with` ステートメント）の利用を推奨しており、run の終了処理が自動的に行われます。

分散トレーニング実験の場合、各プロセスごとに個別の run を作成して追跡するか、全プロセスを 1 つの run でまとめて追跡することができます。詳細は [分散トレーニング実験のログ作成](https://docs.wandb.ai/guides/track/log/distributed-training) をご参照ください。

`wandb.Run.log()` を使って run にデータを記録できます。`wandb.Run.log()` で記録したものは全てその run に送信されます。詳しくは [実験の作成](https://docs.wandb.ai/guides/track/launch) や [`wandb.init`](https://docs.wandb.ai/ref/python/init/) の API リファレンスページをご覧ください。

`Run` オブジェクトは [`wandb.apis.public`](https://docs.wandb.ai/ref/python/public-api/api/) 名前空間にもあり、既に作成されている run に対する操作用です。



**属性 (Attributes):**
 
 - `summary`:  (Summary) run のサマリー情報を格納する辞書型オブジェクト。詳細は
 - [サマリーメトリクスのログ作成](https://docs.wandb.ai/guides/track/log/log-summary/) を参照してください。



**使用例 (Examples):**
`wandb.init()` で run を作成:

```python
import wandb

# 新しい run を開始し、データを記録
# コンテキストマネージャ（with 構文）で自動的に run を終了
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

この run に関連するファイルが保存されるディレクトリ。

---

### <kbd>property</kbd> Run.disabled

run が無効な場合は True、そうでなければ False。

---

### <kbd>property</kbd> Run.entity

この run に紐づく W&B Entity の名前。

Entity はユーザー名、チーム名、または組織名です。

---

### <kbd>property</kbd> Run.group

この run に紐づくグループ名を返します。

run をグループでまとめることで、関連する複数の実験を W&B UI 上でまとめて整理・可視化できます。特に分散トレーニングや交差検証 (cross-validation) といった複数の run を 1 つの実験単位で管理したい場合に便利です。

全プロセスが 1 つの run オブジェクトを共有する「共有モード」では通常グループ設定は不要です。

---

### <kbd>property</kbd> Run.id

この run の識別子（ID）。

---

### <kbd>property</kbd> Run.job_type

この run のジョブタイプ名。

run のジョブタイプは W&B アプリの Overview ページで確認できます。

例えば "training"、"evaluation"、"inference" などのジョブタイプ別に run を分類するのに利用できます。特に様々なタイプの run を扱っている場合、整理・フィルターに便利です。詳細は [run の整理方法](https://docs.wandb.ai/guides/runs/#organize-runs) をご参照ください。

---

### <kbd>property</kbd> Run.name

run の表示名。

表示名に一意性は保証されず、説明的な内容の場合もあります。デフォルトではランダムに生成されます。

---

### <kbd>property</kbd> Run.notes

run に紐づくノート（メモ）が存在する場合、その内容。

ノートは複数行文字列が可能で、markdown や `$$` 内でlatex数式 (`$x + 3$` など) も使用可能です。

---

### <kbd>property</kbd> Run.offline

run がオフラインの場合は True、それ以外は False。

---

### <kbd>property</kbd> Run.path

run へのパス。

パスは entity、project、run ID で構成され、`entity/project/run_id` の形式になります。

---

### <kbd>property</kbd> Run.project

この run に紐づく W&B Project の名称。

---

### <kbd>property</kbd> Run.project_url

この run に紐づく W&B Project の URL（存在する場合）。

オフライン run には project URL はありません。

---

### <kbd>property</kbd> Run.resumed

run が再開された場合は True、そうでなければ False。

---

### <kbd>property</kbd> Run.settings

run の Settings オブジェクトの凍結済みコピー。

---

### <kbd>property</kbd> Run.start_time

run 開始時点の Unix タイムスタンプ（秒）。

---


### <kbd>property</kbd> Run.sweep_id

この run に紐づく sweep の識別子（存在する場合）。

---

### <kbd>property</kbd> Run.sweep_url

この run に紐づく sweep の URL（存在する場合）。

オフライン run には sweep URL はありません。

---

### <kbd>property</kbd> Run.tags

run に紐づけられたタグ（存在する場合）。

---

### <kbd>property</kbd> Run.url

この W&B run の URL（あれば）。

オフライン run には URL はありません。



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

指定されたタイトルとテキストでアラートを作成します。



**引数:**
 
 - `title`:  アラートのタイトル。64 文字未満である必要があります。
 - `text`:  アラートの本文テキスト。
 - `level`:  アラートのレベル。`INFO`, `WARN`, または `ERROR` のいずれか。
 - `wait_duration`:  このタイトルのアラートを再送信するまで待つ時間（秒単位）。

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

`wandb.Run.log()` で記録されるメトリクスをカスタマイズします。



**引数:**
 
 - `name`:  カスタマイズするメトリクスの名前。
 - `step_metric`:  このメトリクスの X 軸となる他のメトリクス名。自動生成グラフで利用します。
 - `step_sync`:  `wandb.Run.log()` で step_metric を明示的に指定しなかった場合、自動で最新値を挿入。step_metric がある場合はデフォルト True。
 - `hidden`:  このメトリクスを自動プロットから非表示にします。
 - `summary`:  summary に追加する集約メトリクス。 min, max, mean, last, best, copy, none から選択。"best" は goal パラメータと併用。 "none" で summary を生成しません。 "copy" は非推奨。
 - `goal`:  "best" summary タイプの解釈方法。"minimize" または "maximize"。
 - `overwrite`:  False の場合、前回の define_metric の未指定内容を引き継ぎます。True の場合は未指定内容も上書きします。



**返り値:**
 この呼び出しを表すオブジェクト（使用しなくても問題ありません）。

---

### <kbd>method</kbd> `Run.display`

```python
display(height: 'int' = 420, hidden: 'bool' = False) → bool
```

この run を Jupyter 上で表示します。

---

### <kbd>method</kbd> `Run.finish`

```python
finish(exit_code: 'int | None' = None, quiet: 'bool | None' = None) → None
```

run を終了し、残りのデータをアップロードします。

W&B run の完了を記録し、全データがサーバーへ同期されることを保証します。run の最終的なステータスは終了条件や同期状況により決まります。

Run の状態:
- Running: データを記録中、またはハートビートを送信中のアクティブな run
- Crashed: ハートビート送信が予期せず停止した run
- Finished: 正常終了（`exit_code=0`）し、全データが同期済み
- Failed: エラー終了（`exit_code!=0`）
- Killed: 強制的に停止された run



**引数:**
 
 - `exit_code`:  run の終了ステータス（整数）。成功時は 0、他は失敗扱い。
 - `quiet`:  非推奨。ログレベルは `wandb.Settings(quiet=...)` で制御してください。

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

未確定の artifact を run のアウトプットとして確定します。

同じ distributed ID で再度 upsert すると新しいバージョンが作成されます。



**引数:**
 
 - `artifact_or_path`:  アーティファクトの内容へのパス。以下いずれか:
            - `/local/directory`
            - `/local/directory/file.txt`
            - `s3://bucket/path`
            - もしくは `wandb.Artifact` で作成した Artifact オブジェクト
 - `name`:  アーティファクト名。`entity/project` 接頭辞があっても OK。有効な名前例:
            - name:version
            - name:alias
            - digest
            - 指定しない場合はパスのベース名 + 現在の run id
 - `type`:  ログする artifact のタイプ。例: `dataset`, `model`
 - `aliases`:  この artifact に適用されるエイリアス。デフォルト `["latest"]`
 - `distributed_id`:  全分散ジョブで共有する一意の文字列。None の場合は run の group 名が使われます。



**返り値:**
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

指定した artifact を portfolio（昇格済み artifact コレクション）へリンクします。

リンクされた artifact は、指定した portfolio の UI で確認できます。



**引数:**
 
 - `artifact`:  リンクする artifact（パブリックまたはローカル）
 - `target_path`:  `str`。以下いずれか: `{portfolio}`, `{project}/{portfolio}`, `{entity}/{project}/{portfolio}`
 - `aliases`:  `List[str]`。この portfolio 内だけで適用されるエイリアス（任意）。 "latest" は常に最新バージョンに自動適用されます。



**返り値:**
 リンクに成功した場合は linked artifact、失敗時は None。

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

モデル artifact のバージョンをログし、Registered Model としてモデルレジストリにリンクします。

リンクしたモデルバージョンは、指定した Registered Model の UI で確認できます。

このメソッドの動作:
- 'name' モデル artifact がすでに存在すれば、`path` のファイルに一致するバージョンを利用。なければ `path` の内容で新規 artifact（タイプ 'model'）を作成。
- 'registered_model_name' でモデルレジストリ上に Registered Model がなければ新規作成。
- 'name' モデル artifact バージョンを Registered Model 'registered_model_name' にリンク。
- 指定した 'aliases' を newly linked artifact version に適用。



**引数:**
 
 - `path`:  (str) モデル内容のパス。例:
    - `/local/directory`
    - `/local/directory/file.txt`
    - `s3://bucket/path`
 - `registered_model_name`:  登録先 Registered Model の名称。チーム固有の ML Task を表すことが多いです。所属 entity は run から取得。
 - `name`:  `path` のファイルを格納するモデル artifact 名（未指定時はパスのベース名 + run id）
 - `aliases`:  登録先 Registered Model 内だけで使うエイリアス。"latest" は常に自動付与されます。



**例外:**
 
 - `AssertionError`:  registered_model_name がパスの場合、または 'name' artifact が "model" を含まないタイプの場合
 - `ValueError`:  name に無効な特殊文字が含まれる場合



**返り値:**
 成功時は linked artifact、失敗時は `None`。

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

このメソッドはスカラー値、画像、動画、ヒストグラム、プロット、テーブルなど各種データをログできます。コード例・ベストプラクティス等は [オブジェクト・メディアのログ作成](https://docs.wandb.ai/guides/track/log) をご覧ください。

基本的な使い方：

```python
import wandb

with wandb.init() as run:
     run.log({"train-loss": 0.5, "accuracy": 0.9})
```

上記コードスニペットは loss と accuracy を run の履歴・summary に保存します。

ログしたデータは [wandb.ai](https://wandb.ai) のワークスペース、[セルフホスト環境](https://docs.wandb.ai/guides/hosting)、もしくは Jupyter notebook 等で [Public API](https://docs.wandb.ai/guides/track/public-api-guide) を利用して可視化・探索できます。

スカラー値以外もログ可能です。[W&B 対応データ型](https://docs.wandb.ai/ref/python/data-types/)（画像や音声、動画など）も記録できます。例えば `wandb.Table` で構造化データを記録可能です。詳しくは [テーブルのログ・可視化・クエリ取得](https://docs.wandb.ai/guides/models/tables/tables-walkthrough) チュートリアルを参照ください。

W&B では `/`（スラッシュ）を含む名前のメトリクスを、スラッシュより前の名前ごとにセクション化します。例えば次の例では "train" と "validate" という 2 つのセクションが出来ます。

```python
with wandb.init() as run:
     # "train" セクションのメトリクスをログ
     run.log(
         {
             "train/accuracy": 0.9,
             "train/loss": 30,
             "validate/accuracy": 0.8,
             "validate/loss": 20,
         }
     )
```

1 階層までのみネスト可能です；`run.log({"a/b/c": 1})` の場合は "a/b" というセクションになります。

`run.log()` は 1 秒間に何回も呼び出す用途には設計されていません。最適なパフォーマンスのためにはログ間隔を調整してください。複数イテレーション分のデータをまとめて記録しても問題ありません。

デフォルトでは、各 `log` 呼び出しが新しい "step" になります。step が自動増分され、前の step へ記録することはできません。どのメトリクスでもチャートの X 軸として利用できます。[カスタムロギング軸](https://docs.wandb.ai/guides/track/log/customize-logging-axes/) も参照ください。

多くの場合、W&B の step はトレーニング step よりもタイムスタンプのように扱うのが推奨されます。

```python
with wandb.init() as run:
     # 例： "epoch" を X 軸用メトリクスとして記録
     run.log({"epoch": 40, "train-loss": 0.5})
```

`step` や `commit` パラメータを使い、同一の step へ複数回の `wandb.Run.log()` 呼び出しも可能です。以下は全て同じ結果になります。

```python
with wandb.init() as run:
     # 通常の使い方
     run.log({"train-loss": 0.5, "accuracy": 0.8})
     run.log({"train-loss": 0.4, "accuracy": 0.9})

     # 自動増分なしの implicit step
     run.log({"train-loss": 0.5}, commit=False)
     run.log({"accuracy": 0.8})
     run.log({"train-loss": 0.4}, commit=False)
     run.log({"accuracy": 0.9})

     # 明示的な step
     run.log({"train-loss": 0.5}, step=current_step)
     run.log({"accuracy": 0.8}, step=current_step)
     current_step += 1
     run.log({"train-loss": 0.4}, step=current_step)
     run.log({"accuracy": 0.9}, step=current_step)
```



**引数:**
 
 - `data`:  `str` をキーとするシリアライズ可能な Python オブジェクトの辞書。
 - `Python objects including`:  `int`、`float`、`string` など、wandb.data_types、またはシリアライズ可能な Python オブジェクトのリスト、タプル、NumPy 配列や、同様の構造の辞書。
 - `step`:  ログ記録する step 番号。指定しない場合は自動で step が増加。詳細は説明欄参照。
 - `commit`:  True で step を確定・アップロード、False で一時的にデータを蓄積。詳細は説明欄参照。`step=None` 時はデフォルト commit=True、それ以外は commit=False。



**使用例:**
 より詳細な例や説明については [ロギングガイド](https://docs.wandb.com/guides/track/log) もご参照ください。

基本的な記録

```python
import wandb

with wandb.init() as run:
    run.log({"train-loss": 0.5, "accuracy": 0.9
``` 

インクリメンタルロギング

```python
import wandb

with wandb.init() as run:
    run.log({"loss": 0.2}, commit=False)
    # 別箇所で step 報告準備ができた時に
    run.log({"accuracy": 0.8})
``` 

ヒストグラム

```python
import numpy as np
import wandb

# 勾配をランダムにサンプリング
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

Matplotlib プロット

```python
from matplotlib import pyplot as plt
import numpy as np
import wandb

with wandb.init() as run:
    fig, ax = plt.subplots()
    x = np.linspace(0, 10)
    y = x * x
    ax.plot(x, y)  # y = x^2 のグラフをプロット
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
 
 - `wandb.Error`:  `wandb.init()` より前に呼び出すとエラー
 - `ValueError`:  無効なデータ渡過時

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

run の出力として artifact を宣言します。



**引数:**
 
 - `artifact_or_path`:  (str または Artifact) artifact の内容へのパス。例:
            - `/local/directory`
            - `/local/directory/file.txt`
            - `s3://bucket/path`
            - `wandb.Artifact` で作成した Artifact オブジェクトも渡せます
 - `name`:  (str, 任意) artifact 名。有効なパターン:
            - name:version
            - name:alias
            - digest
            - 未指定時はパスのベース名 + run id
 - `type`:  (str) artifact のタイプ。例: `dataset`, `model`
 - `aliases`:  (list, 任意) artifact に適用するエイリアス。デフォルトは `["latest"]`
 - `tags`:  (list, 任意) artifact に付与するタグ（任意）



**返り値:**
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

デフォルトでは、カレントディレクトリ内の拡張子 `.py` のファイルを全て記録します。



**引数:**
 
 - `root`:  コード探索の起点となるパス（相対または絶対パス）
 - `name`:  (str, 任意) コード artifact 名。デフォルトは `source-$PROJECT_ID-$ENTRYPOINT_RELPATH` 形式。複数 run で同一 artifact を共有したい場合に有効です。
 - `include_fn`:  ファイルパス（およびルートパス, 任意）を受け取り True/False を返す関数。デフォルトは `lambda path, root: path.endswith(".py")`
 - `exclude_fn`:  ファイルパス（およびルートパス, 任意）を受け取り、対象外の場合 True、それ以外は False を返す関数。デフォルトは `<root>/.wandb/` や `<root>/wandb/` 下を除外します。



**使用例:**
基本的な使い方

```python
import wandb

with wandb.init() as run:
    run.log_code()
```

応用例

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



**返り値:**
 コードがログされた場合は Artifact オブジェクト

---

### <kbd>method</kbd> `Run.log_model`

```python
log_model(
    path: 'StrPath',
    name: 'str | None' = None,
    aliases: 'list[str] | None' = None
) → None
```

`path` 以下のファイルをモデル artifact として記録し、この run のアウトプットにします。

artifact 名には英数字、アンダースコア、ハイフンのみ利用可能です。



**引数:**
 
 - `path`:  (str) モデル内容へのパス。例:
            - `/local/directory`
            - `/local/directory/file.txt`
            - `s3://bucket/path`
 - `name`:  ファイルを格納するモデル artifact 名（未指定時はパスのベース名 + run id）
 - `aliases`:  作成されたモデル artifact につけるエイリアス。デフォルトは `["latest"]`



**例外:**
 
 - `ValueError`:  無効な特殊文字が name に含まれる場合



**返り値:**
 なし

---

### <kbd>method</kbd> `Run.mark_preempting`

```python
mark_preempting() → None
```

この run を「preempting」（割り当て解除前）状態としてマークします。

内部プロセスにも即座にサーバーへレポートするよう指示します。

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

クラウドストレージから指定ファイルをダウンロードします。

ダウンロードはカレントディレクトリ、または run ディレクトリに配置されます。デフォルトではファイルが既存でない場合のみダウンロードを行います。



**引数:**
 
 - `name`:  ファイル名。
 - `run_path`:  取得元 run のパス（例：`username/project_name/run_id`）。`wandb.init` 前の場合は必須。
 - `replace`:  ファイルが既にある場合でもダウンロードするかどうか
 - `root`:  ダウンロード保存先のディレクトリ。デフォルトはカレントディレクトリまたは run ディレクトリ。



**返り値:**
 ファイルが見つからない場合は None、それ以外はリード用のファイルオブジェクト。



**例外:**
 
 - `CommError`:  W&B バックエンドへ接続できない場合
 - `ValueError`:  ファイルが見つからない場合、または run_path が見つからない場合

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

相対パスはカレントディレクトリ基準です。

Unix グロブ（例："myfiles/*"）は `save` 実行時に展開されます（policy に関係なく）。あとから追加したファイルは自動的には拾われません。

アップロード先ディレクトリ構造を制御したい場合、`base_path` を指定することで、`glob_str` の下層構造が preserve されます。

絶対パスやグロブを base_path なしで指定した場合、1 階層上のディレクトリが保持されます。



**引数:**
 
 - `glob_str`:  相対・絶対パス、あるいは Unix グロブ
 - `base_path`:  ディレクトリ構造を推測するためのパス
 - `policy`:  `live`、`now`、`end` のいずれか
    - live: ファイル変更時に随時上書きアップロード
    - now: この呼び出し時のみアップロード
    - end: run 終了時にアップロード



**返り値:**
 マッチしたファイルの symlink へのパス

歴史的理由により、レガシーコードでは bool を返す場合があります。

```python
import wandb

run = wandb.init()

run.save("these/are/myfiles/*")
# => run 上に "these/are/myfiles/" フォルダで保存

run.save("these/are/myfiles/*", base_path="these")
# => run 上では "are/myfiles/" フォルダで保存

run.save("/User/username/Documents/run123/*.txt")
# => run 上で "run123/" フォルダに保存。下部の注意を参照。

run.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => run 上で "username/Documents/run123/" フォルダで保存

run.save("files/*/saveme.txt")
# => "files/" 以下の適切なサブディレクトリにファイルを保存

# コンテキストマネージャ未使用時は明示的に run を終了
run.finish()
```

---

### <kbd>method</kbd> `Run.status`

```python
status() → RunStatus
```

現在の run の同期状況など、内部バックエンドから sync 情報を取得します。

---


### <kbd>method</kbd> `Run.unwatch`

```python
unwatch(
    models: 'torch.nn.Module | Sequence[torch.nn.Module] | None' = None
) → None
```

PyTorch モデルのトポロジー、勾配、パラメータのフックを解除します。



**引数:**
 
 - `models`:  watch を有効にしていた pytorch モデルのリスト（任意）

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

未確定の artifact を run のアウトプットとして宣言（または追記）します。

このメソッドで artifact を作成した後は `run.finish_artifact()` を呼んで確定してください。分散ジョブで複数ジョブが同じ artifact へ貢献する場合などに有用です。



**引数:**
 
 - `artifact_or_path`:  artifact 内容へのパス。例:
    - `/local/directory`
    - `/local/directory/file.txt`
    - `s3://bucket/path`
 - `name`:  artifact 名。"entity/project" で接頭も可。未指定時はパスのベース名 + run ID。可用名例:
    - name:version
    - name:alias
    - digest
 - `type`:  artifact のタイプ。一例として `dataset`, `model`
 - `aliases`:  artifact に適用するエイリアス。デフォルト `["latest"]`
 - `distributed_id`:  分散ジョブで共有する一意の文字列。None の場合は run の group 名



**返り値:**
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

artifact を run の入力として宣言します。

戻り値のオブジェクトに `download` や `file` を呼ぶことで、ローカルで内容を取得できます。



**引数:**
 
 - `artifact_or_name`:  artifact 名。artifact の記録先プロジェクト（`<entity>` または `<entity>/<project>`）をプレフィックスとして指定可能。entity を指定しない場合は Run や API の entity 設定が使われます。有効名例
    - name:version
    - name:alias
 - `type`:  artifact のタイプ
 - `aliases`:  この artifact に適用するエイリアス
 - `use_as`:  非推奨パラメータ（何もしません）



**返り値:**
`Artifact` オブジェクト



**使用例:**
```python
import wandb

run = wandb.init(project="<example>")

# artifact 名/エイリアスで利用
artifact_a = run.use_artifact(artifact_or_name="<name>:<alias>")

# artifact 名/バージョンで利用
artifact_b = run.use_artifact(artifact_or_name="<name>:v<version>")

# entity/project/name:alias を指定
artifact_c = run.use_artifact(
    artifact_or_name="<entity>/<project>/<name>:<alias>"
)

# entity/project/name:version を指定
artifact_d = run.use_artifact(
    artifact_or_name="<entity>/<project>/<name>:v<version>"
)

# コンテキストマネージャ未使用時は明示的に run 終了
run.finish()
```

---

### <kbd>method</kbd> `Run.use_model`

```python
use_model(name: 'str') → FilePathStr
```

指定したモデル artifact 'name' のファイル群をダウンロードします。



**引数:**
 
 - `name`:  モデル artifact 名。既存 artifact 名と一致が必要。`entity/project/` プレフィックス付きでも OK。有効名例:
    - model_artifact_name:version
    - model_artifact_name:alias



**返り値:**
 
 - `path` (str):  ダウンロードしたモデル artifact のファイルパス



**例外:**
 
 - `AssertionError`:  'name' artifact の type に "model" が含まれていない場合

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

指定した PyTorch モデルの勾配や計算グラフを監視できます。

この関数はトレーニング時のパラメータや勾配、またはその両方の追跡が可能です。



**引数:**
 
 - `models`:  監視対象となるモデル、またはモデルのシーケンス
 - `criterion`:  最適化する損失関数（任意）
 - `log`:  "gradients", "parameters", "all" のいずれかを指定。None でロギングを無効化（デフォルトは "gradients"）
 - `log_freq`:  勾配・パラメータの記録頻度（バッチごと、デフォルト: 1000）
 - `idx`:  複数モデル追跡時のインデックス（デフォルト None）
 - `log_graph`:  モデル計算グラフの記録有無（デフォルト False）



**例外:**
 ValueError:  `wandb.init()` 未実行時、または対象が `torch.nn.Module` でない場合
