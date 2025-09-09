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

W&B には Keras 用のコールバックが 3 つあり、`wandb` v0.13.4 から利用できます。レガシーな `WandbCallback` については下へスクロールしてください。


- **`WandbMetricsLogger`** : このコールバックは [実験管理]({{< relref path="/guides/models/track" lang="ja" >}}) に使用します。トレーニングと検証のメトリクスに加えて、システムメトリクスを W&B にログします。

- **`WandbModelCheckpoint`** : このコールバックを使って、モデルのチェックポイントを W&B の [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) にログします。

- **`WandbEvalCallback`**: このベースコールバックは、対話的な可視化のためにモデルの予測を W&B の [Tables]({{< relref path="/guides/models/tables/" lang="ja" >}}) にログします。

これらの新しいコールバックは:

* Keras の設計思想に従います。
* すべてを単一のコールバック（`WandbCallback`）で賄う際の認知的負荷を軽減します。
* Keras ユーザーがサブクラス化してニッチな ユースケース をサポートしやすいように設計されています。

## `WandbMetricsLogger` で 実験 を追跡する

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}

`WandbMetricsLogger` は、`on_epoch_end`、`on_batch_end` などのコールバックメソッドが引数として受け取る Keras の `logs` 辞書を自動でログします。

記録される内容:

* `model.compile` で定義したトレーニングおよび検証メトリクス。
* システム（CPU/GPU/TPU）メトリクス。
* 学習率（固定値でも、学習率スケジューラでも）。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# 新しい W&B Run を初期化
wandb.init(config={"bs": 12})

# WandbMetricsLogger を model.fit に渡す
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbMetricsLogger()]
)
```

### `WandbMetricsLogger` リファレンス


| Parameter | Description | 
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | (`epoch`、`batch`、または `int`)：`epoch` の場合は各エポックの終わりにメトリクスをログします。`batch` の場合は各バッチの終わりにログします。`int` の場合は、その数のバッチごとにログします。デフォルトは `epoch`。                                 |
| `initial_global_step` | (int)：`initial_epoch` から学習を再開し、学習率スケジューラを使用している場合に、学習率を正しくログするために使用します。`step_size * initial_step` として計算できます。デフォルトは 0。 |

## `WandbModelCheckpoint` でモデルをチェックポイント保存する

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}

`WandbModelCheckpoint` コールバックを使うと、Keras モデル（`SavedModel` 形式）またはモデルの重みを一定間隔で保存し、モデルのバージョン管理のために W&B に `wandb.Artifact` としてアップロードできます。 

このコールバックは [`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) を継承しており、チェックポイントのロジックは親コールバックが担当します。

このコールバックで保存できるもの:

* `monitor` に基づいて最高性能を達成したモデル。
* 性能に関係なく各エポック終了時のモデル。
* 各エポックの終わり、または一定数のトレーニングバッチごとに保存されたモデル。
* モデルの重みのみ、またはモデル全体。
* `SavedModel` 形式または `.h5` 形式のモデル。

このコールバックは `WandbMetricsLogger` と併用してください。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# 新しい W&B Run を初期化
wandb.init(config={"bs": 12})

# WandbModelCheckpoint を model.fit に渡す
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

| Parameter | Description | 
| ------------------------- |  ---- | 
| `filepath`   | (str)：モデルファイルの保存先パス。|  
| `monitor`                 | (str)：監視するメトリクス名。         |
| `verbose`                 | (int)：冗長度。0 または 1。0 は無出力、1 はコールバックが動作したときにメッセージを表示。   |
| `save_best_only`          | (Boolean)：`save_best_only=True` の場合、`monitor` と `mode` で定義された条件に従って、最新のモデルまたは最良と判断されたモデルのみを保存します。   |
| `save_weights_only`       | (Boolean)：True の場合、モデルの重みのみを保存します。                                            |
| `mode`                    | (`auto`、`min`、`max`)：`val_acc` には `max`、`val_loss` には `min`、など。  |                     |
| `save_freq`               | ("epoch" または int)：`epoch` の場合、各エポック後にモデルを保存。整数の場合、その数のバッチごとに保存。`val_acc` や `val_loss` などの検証メトリクスを監視する場合、それらはエポックの終わりにしか得られないため、`save_freq` は "epoch" に設定する必要があります。 |
| `options`                 | (str)：`save_weights_only` が true の場合は任意の `tf.train.CheckpointOptions` オブジェクト、false の場合は任意の `tf.saved_model.SaveOptions` オブジェクト。    |
| `initial_value_threshold` | (float)：監視対象メトリクスの初期「ベスト」値。       |

### N エポックごとにチェックポイントをログする

デフォルト（`save_freq="epoch"`）では、各エポック後にチェックポイントを作成し、Artifact としてアップロードします。特定のバッチ数ごとにチェックポイントを作成するには、`save_freq` を整数に設定します。`N` エポックごとにチェックポイントしたい場合は、`train` データローダのカーディナリティを計算し、それを `save_freq` に渡します:

```python
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### TPU アーキテクチャーで効率的にチェックポイントをログする

TPU でチェックポイント保存を行うと、`UnimplementedError: File system scheme '[local]' not implemented` というエラーメッセージに遭遇することがあります。これは、モデルディレクトリー（`filepath`）にクラウドストレージのバケットパス（`gs://bucket-name/...`）を使用する必要があり、そのバケットに TPU サーバーから アクセス できる必要があるためです。とはいえ、ローカルパスでチェックポイントを作成し、その後 Artifacts としてアップロードする方法も使えます。

```python
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## `WandbEvalCallback` でモデル予測を可視化する

{{< cta-button colabLink="https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}

`WandbEvalCallback` は、主にモデルの予測、次いでデータセットの可視化のために Keras コールバックを構築するための抽象基底クラスです。

この抽象コールバックはデータセットやタスクに依存しません。使用するには、このベースの `WandbEvalCallback` を継承し、`add_ground_truth` と `add_model_prediction` メソッドを実装してください。

`WandbEvalCallback` は、次のメソッドを提供するユーティリティクラスです:

* データと予測の `wandb.Table` インスタンスを作成。
* データと予測の Tables を `wandb.Artifact` としてログ。
* `on_train_begin` でデータテーブルをログ。
* `on_epoch_end` で予測テーブルをログ。

次の例では、画像分類タスクに `WandbClfEvalCallback` を使用します。このコールバックは検証データ（`data_table`）を W&B にログし、推論を実行し、各エポックの終わりに予測（`pred_table`）を W&B にログします。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback


# モデルの予測可視化用コールバックを実装
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

# 新しい W&B Run を初期化
wandb.init(config={"hyper": "parameter"})

# Callbacks を model.fit に追加
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
W&B の [Artifact page]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ja" >}}) には、デフォルトで Table のログが含まれます。**Workspace** ページではありません。
{{% /alert %}}

### `WandbEvalCallback` リファレンス

| Parameter            | Description                                      |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (list) `data_table` の列名リスト |
| `pred_table_columns` | (list) `pred_table` の列名リスト |

### メモリ使用量の詳細

`on_train_begin` が呼ばれたときに `data_table` を W&B にログします。W&B の Artifact としてアップロードされた後は、このテーブルへの参照を取得し、`data_table_ref` クラス変数から アクセス できます。`data_table_ref` は 2 次元リストで、`self.data_table_ref[idx][n]` のようにインデックスできます。ここで `idx` は行番号、`n` は列番号です。以下の例で使い方を確認しましょう。

### コールバックをカスタマイズする

より細かな制御のために、`on_train_begin` や `on_epoch_end` をオーバーライドできます。サンプルを `N` バッチごとにログしたい場合は、`on_train_batch_end` メソッドを実装してください。

{{% alert %}}
`WandbEvalCallback` を継承してモデル予測の可視化用コールバックを実装している際に、不明点や改善点があれば、[issue](https://github.com/wandb/wandb/issues) を作成してお知らせください。
{{% /alert %}}

## `WandbCallback` [legacy]

W&B ライブラリの `WandbCallback` クラスを使うと、`model.fit` で追跡されるすべてのメトリクスと損失値を自動的に保存できます。

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Keras でモデルをセットアップするコード

# コールバックを model.fit に渡す
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

1 分以内でできる Keras と W&B の入門動画は [Get Started with Keras and W&B in Less Than a Minute](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M) をご覧ください。

より詳しい解説は [Integrate W&B with Keras](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab_channel=Weights%26Biases) を参照してください。[Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras_pipeline_with_Weights_and_Biases.ipynb) も確認できます。

{{% alert %}}
スクリプトは [example repo](https://github.com/wandb/examples) をご覧ください。[Fashion MNIST example](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py) や、そこから生成される [W&B ダッシュボード](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) も含まれています。
{{% /alert %}}

`WandbCallback` クラスは幅広いログ設定オプションをサポートしています。監視するメトリクスの指定、重みや勾配の追跡、`training_data` や `validation_data` 上での予測のログなどです。

詳細は `keras.WandbCallback` のリファレンスドキュメントをご確認ください。

`WandbCallback` は次のことを行います:

* Keras が収集したあらゆるメトリクス（損失および `keras_model.compile()` に渡したもの）から履歴データを自動でログします。
* `monitor` と `mode` によって定義される「最良」のトレーニングステップに対応する Run にサマリーメトリクスを設定します。デフォルトでは `val_loss` が最小のエポックです。`WandbCallback` はデフォルトで、最良の `epoch` に対応するモデルを保存します。
* 任意で勾配とパラメータのヒストグラムをログします。
* 任意で、W&B が可視化できるようにトレーニングおよび検証データを保存します。

### `WandbCallback` リファレンス

| Arguments                  |                                    |
| -------------------------- | ------------------------------------------- |
| `monitor`                  | (str) 監視するメトリクス名。デフォルトは `val_loss`。                                                                   |
| `mode`                     | (str) {`auto`、`min`、`max`} のいずれか。`min` - monitor が最小化されたときにモデルを保存。`max` - monitor が最大化されたときにモデルを保存。`auto` - いつ保存するかを自動推測（デフォルト）。                                                                                                                                                |
| `save_model`               | True - monitor がこれまでのエポックを上回ったときにモデルを保存。False - モデルを保存しない。                                       |
| `save_graph`               | (boolean) True の場合、モデルグラフを wandb に保存（デフォルト True）。                                                           |
| `save_weights_only`        | (boolean) True の場合はモデルの重みのみを保存（`model.save_weights(filepath)`）。それ以外はモデル全体を保存。   |
| `log_weights`              | (boolean) True の場合、各レイヤーの重みのヒストグラムを保存。                                                |
| `log_gradients`            | (boolean) True の場合、学習時の勾配のヒストグラムをログ。                                                       |
| `training_data`            | (tuple) `model.fit` に渡す `(X, y)` と同じ形式。勾配計算に必要です。`log_gradients` が `True` の場合は必須。       |
| `validation_data`          | (tuple) `model.fit` に渡す `(X, y)` と同じ形式。W&B が可視化するためのデータセット。これを設定すると、毎エポック、少数の予測を実行して結果を保存し、後で可視化できます。          |
| `generator`                | (generator) W&B が可視化するための検証データを返すジェネレータ。`(X, y)` のタプルを返す必要があります。特定のデータ例を可視化するには、`validate_data` または generator のいずれかを設定します。     |
| `validation_steps`         | (int) `validation_data` がジェネレータの場合、検証セット全体で何ステップ実行するか。       |
| `labels`                   | (list) データを W&B で可視化する場合、複数クラス分類のときに数値出力をわかりやすい文字列に変換するラベルリスト。2 値分類では、2 つのラベル \[`label for false`, `label for true`] を渡します。`validate_data` と `generator` が両方とも False の場合は無効。    |
| `predictions`              | (int) 各エポックで可視化のために行う予測数。最大 100。    |
| `input_type`               | (string) 可視化を助けるためのモデル入力タイプ。次のいずれか：(`image`、`images`、`segmentation_mask`)。  |
| `output_type`              | (string) 可視化を助けるためのモデル出力タイプ。次のいずれか：(`image`、`images`、`segmentation_mask`)。    |
| `log_evaluation`           | (boolean) True の場合、各エポックの検証データとモデルの予測を含む Table を保存。詳細は `validation_indexes`、`validation_row_processor`、`output_row_processor` を参照。     |
| `class_colors`             | (\[float, float, float]) 入力または出力がセグメンテーションマスクの場合、各クラスの RGB タプル（0〜1 の範囲）を格納する配列。                  |
| `log_batch_frequency`      | (integer) None の場合、毎エポックでログ。整数を指定すると、トレーニングメトリクスを `log_batch_frequency` バッチごとにログ。          |
| `log_best_prefix`          | (string) None の場合、追加のサマリーメトリクスは保存しません。文字列を指定すると、その接頭辞を監視メトリクスとエポックに付けてサマリーとして保存。   |
| `validation_indexes`       | (\[wandb.data_types._TableLinkMixin]) 各検証サンプルに関連付けるインデックスキーの順序付きリスト。`log_evaluation` が True で `validation_indexes` を指定した場合、検証データの Table は作成せず、各予測を `TableLinkMixin` が表す行に関連付けます。行キーの取得には `Table.get_index()` を使用。        |
| `validation_row_processor` | (Callable) 検証データに適用する関数。一般的にはデータの可視化に使用。関数は `ndx`（int）と `row`（dict）を受け取ります。モデル入力が単一の場合は `row["input"]` に入力データが、複数入力の場合は入力スロット名が入ります。fit のターゲットが単一なら `row["target"]` にターゲットデータ、複数なら出力スロット名が入ります。例：入力データが単一配列で、画像として可視化したい場合は `lambda ndx, row: {"img": wandb.Image(row["input"])}` を指定。`log_evaluation` が False または `validation_indexes` がある場合は無視。 |
| `output_row_processor`     | (Callable) `validation_row_processor` と同様ですが、モデル出力に適用します。`row["output"]` にモデル出力が入ります。          |
| `infer_missing_processors` | (Boolean) `validation_row_processor` と `output_row_processor` が無い場合に推論するかどうか。デフォルトは True。`labels` を指定すると、W&B は適宜分類用のプロセッサを推論します。      |
| `log_evaluation_frequency` | (int) 評価結果をどの頻度でログするか。デフォルト `0` は学習終了時のみ。1 なら毎エポック、2 なら 1 つおきのエポック、というように設定。`log_evaluation` が False の場合は無効。    |

## よくある質問

### `wandb` と一緒に `Keras` のマルチプロセッシングを使うには？

`use_multiprocessing=True` を設定すると、次のエラーが発生することがあります:

```python
Error("You must call wandb.init() before wandb.config.batch_size")
```

回避方法:

1. `Sequence` クラスの構築時に `wandb.init(group='...')` を追加します。
2. `main` 内で `if __name__ == "__main__":` を使い、スクリプトの残りのロジックをその中に入れてください。