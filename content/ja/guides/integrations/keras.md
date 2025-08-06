---
title: Keras
menu:
  default:
    identifier: ja-guides-integrations-keras
    parent: integrations
weight: 160
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases_keras.ipynb" >}}

## Keras コールバック

W&B には Keras 用の 3 つのコールバックが用意されています（`wandb` v0.13.4 から利用可能）。従来の `WandbCallback` については下の方に記載しています。


- **`WandbMetricsLogger`** : このコールバックは [実験管理]({{< relref path="/guides/models/track" lang="ja" >}}) 用に使います。トレーニング・検証メトリクスとシステムメトリクスを W&B に記録します。

- **`WandbModelCheckpoint`** : モデルのチェックポイントを W&B [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) に記録するためのコールバックです。

- **`WandbEvalCallback`**: このベースコールバックは、モデルの予測を W&B [Tables]({{< relref path="/guides/models/tables/" lang="ja" >}}) に記録し、インタラクティブに可視化できます。

新しいコールバックの特徴：

* Keras の設計思想に沿っています。
* すべてを 1 つのコールバック（`WandbCallback`）で済ませる場合に比べ、使いやすさと認知負荷を低減します。
* Keras ユーザーがサブクラス化して独自のユースケースにあわせて簡単にカスタマイズできます。

## `WandbMetricsLogger` で実験管理

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}

`WandbMetricsLogger` は、Keras のコールバックメソッド（`on_epoch_end`, `on_batch_end` など）が引数として受け取る `logs` 辞書を自動的に記録します。

このコールバックで記録されるもの：

* `model.compile` で定義されたトレーニング・検証メトリクス
* システム（CPU/GPU/TPU）メトリクス
* 学習率（固定値もしくは学習率スケジューラのいずれにも対応）

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# 新規 W&B Run を初期化
wandb.init(config={"bs": 12})

# model.fit に WandbMetricsLogger を渡す
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbMetricsLogger()]
)
```

### `WandbMetricsLogger` リファレンス


| パラメータ | 説明 | 
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | (`epoch`, `batch`, または `int`): `epoch` の場合は各エポック終了時、`batch` の場合は各バッチ終了時にメトリクスを記録。数値を指定した場合はそのバッチ数ごとにログされます。デフォルトは `epoch`。                              |
| `initial_global_step` | (int): 一部の初期エポックからトレーニング再開時、正しく学習率を記録するための引数。`step_size * initial_step` で計算できます。デフォルトは 0。 |

## `WandbModelCheckpoint` でモデルのチェックポイントを記録

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}

`WandbModelCheckpoint` コールバックを使うことで、Keras モデル（`SavedModel` 形式）や重みを定期的に保存し、W&B の `wandb.Artifact` としてアップロードしてモデルのバージョン管理ができます。

このコールバックは [`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) から派生しており、チェックポイントの保存ロジックは親コールバックで管理されています。

このコールバックで保存できるもの：

* モニターしているメトリクスでベストなパフォーマンスを達成したモデル
* パフォーマンスに関わらず全エポック終了時のモデル
* エポックの最後／指定したバッチ数ごとに保存するモデル
* モデル全体または重みのみ
* `SavedModel` もしくは `.h5` 形式での保存

`WandbMetricsLogger` と併用してください。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# 新規 W&B Run を初期化
wandb.init(config={"bs": 12})

# model.fit に WandbModelCheckpoint を渡す
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[
        WandbMetricsLogger(),
        WandbModelCheckpoint("models"),
    ],
)
```

### `WandbModelCheckpoint` リファレンス

| パラメータ | 説明 | 
| ------------------------- |  ---- | 
| `filepath`   | (str): モデルファイルを保存するパス |  
| `monitor`                 | (str): モニター対象となるメトリクス名       |
| `verbose`                 | (int): ログの出力モード。0 は停止、1 はコールバックが動作したときにメッセージを表示   |
| `save_best_only`          | (Boolean): `True` の場合、`monitor` と `mode` で定義された条件で最新またはベストなモデルのみを保存   |
| `save_weights_only`       | (Boolean): `True` の場合、モデルの重みのみを保存                             |
| `mode`                    | (`auto`, `min`, `max`): `val_acc` の場合は `max`、`val_loss` の場合は `min` など                       |
| `save_freq`               | ("epoch" または int): `epoch` なら各エポック後、数値ならそのバッチ数ごとにモデルを保存。`val_acc` や `val_loss` のような検証メトリクスをモニターする場合は必ず "epoch" を設定してください。   |
| `options`                 | (str): `save_weights_only` が true の場合は `tf.train.CheckpointOptions`、false の場合は `tf.saved_model.SaveOptions` のオプションオブジェクト（任意）    |
| `initial_value_threshold` | (float): 監視対象メトリクスの初期 "ベスト" 値を浮動小数点数で指定         |

### N エポックごとにチェックポイントを記録

デフォルト（`save_freq="epoch"`）では、各エポック後にチェックポイントを作成しアーティファクトとしてアップロードします。特定のバッチ数ごとにチェックポイントを作成したい場合は `save_freq` を整数に設定してください。`N` エポックごとにチェックポイントを作成したい場合は、`train` データローダーの全件数を取得して `save_freq` に渡します。

```python
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### TPU アーキテクチャで効率良くチェックポイントを記録

TPU 上でチェックポイントを作成する際に `UnimplementedError: File system scheme '[local]' not implemented` というエラーが発生することがあります。これはモデルディレクトリー（`filepath`）がクラウドストレージバケットパス（`gs://bucket-name/...`）である必要があり、そのバケットが TPU サーバーからアクセス可能でなければならないためです。ただし、一旦ローカルパスでチェックポイントを作成し、その後 Artifacts 経由でアップロードすることもできます。

```python
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## `WandbEvalCallback` でモデル予測を可視化

{{< cta-button colabLink="https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}

`WandbEvalCallback` は、主にモデル予測とデータセットの可視化のために設計された Keras コールバックの抽象基底クラスです。

この抽象コールバックは、データセットやタスクに依存しません。使うためにはこのベースクラスを継承し、`add_ground_truth` と `add_model_prediction` メソッドを実装してください。

`WandbEvalCallback` は以下のユーティリティ機能を提供します：

* データと予測の `wandb.Table` インスタンスを作成
* データ・予測の Table を `wandb.Artifact` として記録
* `on_train_begin` でデータテーブルをログ
* `on_epoch_end` で予測テーブルをログ

下記は画像分類タスク用に `WandbClfEvalCallback` を使う例です。このコールバックは検証データ（`data_table`）を W&B に記録し、推論を行いエポック終了ごとに予測結果（`pred_table`）を記録します。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback

# モデル予測可視化用のコールバックを実装
class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(
        self, validation_data, data_table_columns, pred_table_columns, num_samples=100
    ):
        super().__init__(data_table_columns, pred_table_columns)

        self.x = validation_data[0]
        self.y = validation_data[1]

    def add_ground_truth(self, logs=None):
        for idx, (image, label) in enumerate(zip(self.x, self.y)):
            self.data_table.add_data(idx, wandb.Image(image), label)

    def add_model_predictions(self, epoch, logs=None):
        preds = self.model.predict(self.x, verbose=0)
        preds = tf.argmax(preds, axis=-1)

        table_idxs = self.data_table_ref.get_index()

        for idx in table_idxs:
            pred = preds[idx]
            self.pred_table.add_data(
                epoch,
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                self.data_table_ref.data[idx][2],
                pred,
            )

# ...

# 新規 W&B Run を初期化
wandb.init(config={"hyper": "parameter"})

# Model.fit にコールバックを追加
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[
        WandbMetricsLogger(),
        WandbClfEvalCallback(
            validation_data=(X_test, y_test),
            data_table_columns=["idx", "image", "label"],
            pred_table_columns=["epoch", "idx", "image", "label", "pred"],
        ),
    ],
)
```

{{% alert %}}
W&B の [Artifact ページ]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ja" >}}) には Table ログがデフォルトで表示されます。**Workspace** ページではありません。
{{% /alert %}}

### `WandbEvalCallback` リファレンス

| パラメータ              | 説明                                      |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (list) `data_table` のカラム名リスト |
| `pred_table_columns` | (list) `pred_table` のカラム名リスト |

### メモリフットプリントの詳細

`on_train_begin` が呼ばれたタイミングで `data_table` を W&B に記録します。アップロードが完了すると、このテーブルへの参照（`data_table_ref` クラス変数）を取得できます。`data_table_ref` は 2 次元リストで、`self.data_table_ref[idx][n]` の形でインデックスできます。`idx` は行番号、`n` は列番号です。例の使い方を参照してください。

### コールバックのカスタマイズ

必要に応じて `on_train_begin` や `on_epoch_end` メソッドをオーバーライドして、より細かい制御が可能です。`N` バッチごとにサンプルを記録したい場合は `on_train_batch_end` メソッドを実装できます。

{{% alert %}}
`WandbEvalCallback` を継承してモデル予測可視化用のコールバックを実装している際に、不明点や修正要望がある場合は [issue](https://github.com/wandb/wandb/issues) をオープンしてください。
{{% /alert %}}

## `WandbCallback` 【レガシー】

W&B ライブラリの `WandbCallback` クラスを使うと、`model.fit` で追跡された全メトリクスやロス値を自動で記録できます。

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Keras でモデルのセットアップ

# Callbacks に渡す
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

[Get Started with Keras and W&B in Less Than a Minute](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M)（1 分以内で始められる Keras × W&B の動画）もご覧ください。

より詳しい動画は [Integrate W&B with Keras](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab_channel=Weights%26Biases) をご確認ください。[Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras_pipeline_with_Weights_and_Biases.ipynb) も用意されています。

{{% alert %}}
[example repo](https://github.com/wandb/examples) ではスクリプト例（[Fashion MNIST example](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py) など）や、その結果生成される [W&B ダッシュボード](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) も見られます。
{{% /alert %}}

`WandbCallback` クラスは多様なロギング設定オプションに対応：モニタリングするメトリクス指定、重み・勾配の追跡、トレーニングデータと検証データでの予測ロギングなど。

`keras.WandbCallback` の詳細はリファレンス文書をご覧ください。

`WandbCallback` の主な特徴：

* Keras で収集されたすべてのメトリクス（ロスや `keras_model.compile()` で指定したもの）を自動で記録
* `monitor` と `mode` 属性で定義された「最も良い」トレーニングステップに紐付くサマリーメトリクスを設定（デフォルトは最小の `val_loss` を持つエポック、ベストな `epoch` のモデルが保存されます）
* オプションで勾配やパラメータのヒストグラムも記録可能
* オプションでトレーニング・検証データも wandb で可視化用に保存

### `WandbCallback` リファレンス

| 引数                         |                                    |
| --------------------------- | ------------------------------------------- |
| `monitor`                  | (str) モニターするメトリクス名。デフォルトは `val_loss`。 |
| `mode`                     | (str) `{auto, min, max}` のいずれか。`min` - monitor が最小化されるとき保存、`max` - monitor が最大化されるとき保存、`auto` - 自動検出（デフォルト） |
| `save_model`               | `True` で monitor が過去最高になったときにモデル保存、`False` で保存しない        |
| `save_graph`               | (boolean) `True` でモデルグラフを wandb に保存（デフォルトは True）            |
| `save_weights_only`        | (boolean) `True` なら重みのみを保存（`model.save_weights(filepath)`）、`False` ならモデル全体を保存      |
| `log_weights`              | (boolean) `True` で各レイヤーの重みのヒストグラムを保存                    |
| `log_gradients`            | (boolean) `True` でトレーニング勾配のヒストグラムを記録                    |
| `training_data`            | (tuple) `model.fit` で渡す `(X,y)` と同じ形式。勾配計算のため必須――`log_gradients` が `True` の場合は必須         |
| `validation_data`          | (tuple) `model.fit` で渡す `(X,y)` と同じ検証データ。これを指定すると各エポックごとに少数サンプルで予測してあとで可視化できるよう記録                    |
| `generator`                | (generator) wandb で可視化する検証データを返すジェネレータ。`(X,y)` のタプルを返す必要あり。`validate_data` または generator のどちらかが設定されている必要があります         |
| `validation_steps`         | (int) `validation_data` がジェネレータの場合、全検証セットをカバーするステップ数            |
| `labels`                   | (list) データ可視化時に数値出力をわかりやすい文字列に変換するラベルリスト（多クラス分類用）。バイナリ分類なら `[False 用ラベル, True 用ラベル]`。`validate_data`・generator のいずれも設定がない場合は無効            |
| `predictions`              | (int) 各エポックで可視化のために記録する予測サンプル数（最大 100）       |
| `input_type`               | (string) 可視化時のモデル入力タイプ（`image`, `images`, `segmentation_mask` のいずれか）             |
| `output_type`              | (string) 可視化時の出力タイプ（`image`, `images`, `segmentation_mask` のいずれか）                 |
| `log_evaluation`           | (boolean) 各エポックごとに検証データとモデル予測の Table を保存するか。詳細は `validation_indexes`, `validation_row_processor`, `output_row_processor` を参照 |
| `class_colors`             | ([float, float, float]) 入出力がセグメンテーションマスクの場合、各クラスの RGB タプル（0-1 範囲）のリスト    |
| `log_batch_frequency`      | (integer) None ならエポックごと、整数を与えるとそのバッチ数ごとにトレーニングメトリクスを記録          |
| `log_best_prefix`          | (string) None なら追加サマリーメトリクスなし。文字列指定時は prefix を付与して監視メトリクスやエポックをサマリーメトリクスに保存 |
| `validation_indexes`       | ([wandb.data_types._TableLinkMixin]) 各検証サンプルに割り当てるインデックスキーの順序付きリスト。`log_evaluation=True` かつ指定時は Table を作らず、このキーで各予測を関連付ける。行キー取得は `Table.get_index()` を参照                 |
| `validation_row_processor` | (Callable) 検証データに適用する関数。データ可視化によく使われ、引数は `ndx` (int) と `row` (dict)。入力が1つなら `row["input"]`，ターゲットが1つなら `row["target"]`。例えば画像として可視化したければ `lambda ndx, row: {"img": wandb.Image(row["input"])}` のように渡す。`log_evaluation=False` または `validation_indexes` がある場合は無効化される           |
| `output_row_processor`     | (Callable) 検証データと同様だが、モデル出力の可視化用（`row["output"]` には出力結果が格納される）   |
| `infer_missing_processors` | (Boolean) `validation_row_processor` や `output_row_processor` が未設定時は自動補完するか。デフォルト True、`labels` があれば分類用プロセッサも自動推定          |
| `log_evaluation_frequency` | (int) 評価結果の記録頻度。デフォルトは `0`（トレーニング終了時のみ記録）、1 で毎エポックごと、2 で1つおきのエポックなど。`log_evaluation=False` の場合は意味なし        |

## よくある質問

### `Keras` の multiprocessing を wandb と併用するには？

`use_multiprocessing=True` を指定したとき、次のエラーが発生することがあります：

```python
Error("You must call wandb.init() before wandb.config.batch_size")
```

この問題を回避するには：

1. `Sequence` クラスのコンストラクタ内に `wandb.init(group='...')` を追加してください。
2. `main` では `if __name__ == "__main__":` でガードし、その中にスクリプトの処理を記載してください。
