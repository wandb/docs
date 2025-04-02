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

W&B には Keras 用の 3 つのコールバックがあり、`wandb` v0.13.4 から利用できます。従来の `WandbCallback` については、下にスクロールしてください。

- **`WandbMetricsLogger`**: [実験管理]({{< relref path="/guides/models/track" lang="ja" >}})には、このコールバックを使用します。トレーニングと検証のメトリクスをシステムメトリクスとともに Weights & Biases に記録します。

- **`WandbModelCheckpoint`**: モデルのチェックポイントを Weight and Biases [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}})に記録するには、このコールバックを使用します。

- **`WandbEvalCallback`**: このベースコールバックは、モデルの予測を Weights and Biases [Tables]({{< relref path="/guides/models/tables/" lang="ja" >}})に記録して、インタラクティブな可視化を実現します。

これらの新しいコールバック：

* Keras の設計理念に準拠しています。
* すべてに単一のコールバック（`WandbCallback`）を使用することによる認知負荷を軽減します。
* Keras ユーザーがコールバックをサブクラス化してニッチなユースケースをサポートすることで、コールバックを簡単に変更できるようにします。

## `WandbMetricsLogger` で実験を追跡する

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}

`WandbMetricsLogger` は、`on_epoch_end`、`on_batch_end` などのコールバックメソッドが引数として受け取る Keras の `logs` 辞書を自動的に記録します。

これにより、以下が追跡されます。

* `model.compile` で定義されたトレーニングおよび検証のメトリクス。
* システム (CPU/GPU/TPU) メトリクス。
* 学習率 (固定値と学習率スケジューラの両方)。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# 新しい W&B の run を初期化します。
wandb.init(config={"bs": 12})

# WandbMetricsLogger を model.fit に渡します。
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbMetricsLogger()]
)
```

### `WandbMetricsLogger` リファレンス

| パラメータ | 説明 |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | (`epoch`、`batch`、または `int`): `epoch` の場合、各エポックの最後にメトリクスを記録します。`batch` の場合、各バッチの最後にメトリクスを記録します。`int` の場合、その数のバッチの最後にメトリクスを記録します。デフォルトは `epoch` です。                                 |
| `initial_global_step` | (int): 学習率スケジューラを使用している場合に、いくつかの initial_epoch からトレーニングを再開するときに学習率を正しく記録するには、この引数を使用します。これは step_size * initial_step として計算できます。デフォルトは 0 です。 |

## `WandbModelCheckpoint` を使用してモデルをチェックポイントする

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}

`WandbModelCheckpoint` コールバックを使用して、Keras モデル (`SavedModel` 形式) またはモデルの重みを定期的に保存し、モデルの バージョン管理 用の `wandb.Artifact` として W&B にアップロードします。

このコールバックは [`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) からサブクラス化されているため、チェックポイントロジックは親コールバックによって処理されます。

このコールバックは以下を保存します。

* モニターに基づいて最高のパフォーマンスを達成したモデル。
* パフォーマンスに関係なく、すべてのエポックの終わりにモデル。
* エポックの終わり、または固定数のトレーニングバッチの後。
* モデルの重みのみ、またはモデル全体。
* `SavedModel` 形式または `.h5` 形式のいずれかでモデル。

このコールバックを `WandbMetricsLogger` と組み合わせて使用​​します。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# 新しい W&B の run を初期化します。
wandb.init(config={"bs": 12})

# WandbModelCheckpoint を model.fit に渡します。
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
| `filepath`   | (str): モードファイルを保存するパス。|
| `monitor`                 | (str): 監視するメトリック名。|
| `verbose`                 | (int): 詳細モード、0 または 1。モード 0 はサイレントで、モード 1 はコールバックがアクションを実行するときにメッセージを表示します。|
| `save_best_only`          | (Boolean): `save_best_only=True` の場合、最新のモデル、または `monitor` および `mode` 属性で定義されている、最良と見なされるモデルのみを保存します。|
| `save_weights_only`       | (Boolean): True の場合、モデルの重みのみを保存します。|
| `mode`                    | (`auto`、`min`、または `max`): `val_acc` の場合は `max` に、`val_loss` の場合は `min` に設定します。|                     |
| `save_freq`               | ("epoch" または int): 「epoch」を使用する場合、コールバックは各エポックの後にモデルを保存します。整数を使用する場合、コールバックはこの数のバッチの終わりにモデルを保存します。`val_acc` や `val_loss` などの検証メトリクスを監視する場合、これらのメトリクスはエポックの最後にのみ使用できるため、`save_freq` を "epoch" に設定する必要があることに注意してください。 |
| `options`                 | (str): `save_weights_only` が true の場合はオプションの `tf.train.CheckpointOptions` オブジェクト、`save_weights_only` が false の場合はオプションの `tf.saved_model.SaveOptions` オブジェクト。|
| `initial_value_threshold` | (float): 監視するメトリックの浮動小数点初期「最良」値。|

### N エポック後にチェックポイントを記録する

デフォルト (`save_freq="epoch"`) では、コールバックは各エポックの後にチェックポイントを作成し、アーティファクトとしてアップロードします。特定の数のバッチの後にチェックポイントを作成するには、`save_freq` を整数に設定します。`N` エポック後にチェックポイントを作成するには、`train` データローダーのカーディナリティを計算し、それを `save_freq` に渡します。

```python
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### TPU アーキテクチャでチェックポイントを効率的に記録する

TPU でチェックポイントを作成しているときに、`UnimplementedError: File system scheme '[local]' not implemented` エラーメッセージが表示される場合があります。これは、モデルディレクトリー (`filepath`) がクラウドストレージバケットパス (`gs://bucket-name/...`) を使用する必要があり、このバケットが TPU サーバーからアクセスできる必要があるために発生します。ただし、チェックポイント作成にはローカルパスを使用できます。これは Artifacts としてアップロードされます。

```python
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## `WandbEvalCallback` を使用してモデルの予測を可視化する

{{< cta-button colabLink="https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}

`WandbEvalCallback` は、主にモデルの予測、次にデータセットの可視化のための Keras コールバックを構築するための抽象ベースクラスです。

この抽象コールバックは、データセットとタスクに関して不可知論的です。これを使用するには、このベースの `WandbEvalCallback` コールバッククラスから継承し、`add_ground_truth` メソッドと `add_model_prediction` メソッドを実装します。

`WandbEvalCallback` は、次のメソッドを提供するユーティリティクラスです。

* データと予測の `wandb.Table` インスタンスを作成します。
* データと予測の Tables を `wandb.Artifact` として記録します。
* データテーブル `on_train_begin` を記録します。
* 予測テーブル `on_epoch_end` を記録します。

次の例では、画像分類タスクに `WandbClfEvalCallback` を使用しています。この例のコールバックは、検証データ (`data_table`) を W&B に記録し、推論を実行し、すべてのエポックの終わりに予測 (`pred_table`) を W&B に記録します。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback


# モデル予測の可視化コールバックを実装します。
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

# 新しい W&B の run を初期化します。
wandb.init(config={"hyper": "parameter"})

# コールバックを Model.fit に追加します。
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
W&B の [Artifact page]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ja" >}})には、デフォルトで **Workspace** ページではなく、Table ログが含まれています。
{{% /alert %}}

### `WandbEvalCallback` リファレンス

| パラメータ | 説明 |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (list) `data_table` の列名のリスト |
| `pred_table_columns` | (list) `pred_table` の列名のリスト |

### メモリフットプリントの詳細

`on_train_begin` メソッドが呼び出されると、`data_table` を W&B に記録します。W&B Artifact としてアップロードされると、`data_table_ref` クラス変数を使用してアクセスできるこのテーブルへの参照を取得します。`data_table_ref` は、`self.data_table_ref[idx][n]` のようにインデックスを付けることができる 2D リストです。ここで、`idx` は行番号で、`n` は列番号です。以下の例で使用法を見てみましょう。

### コールバックをカスタマイズする

`on_train_begin` メソッドまたは `on_epoch_end` メソッドをオーバーライドして、よりきめ細かい制御を行うことができます。`N` バッチ後にサンプルを記録する場合は、`on_train_batch_end` メソッドを実装できます。

{{% alert %}}
💡 `WandbEvalCallback` を継承してモデル予測の可視化のためのコールバックを実装していて、明確にする必要がある場合や修正する必要がある場合は、[issue](https://github.com/wandb/wandb/issues) を開いてお知らせください。
{{% /alert %}}

## `WandbCallback` [レガシー]

W&B ライブラリの [`WandbCallback`]({{< relref path="/ref/python/integrations/keras/wandbcallback" lang="ja" >}}) クラスを使用して、`model.fit` で追跡されるすべてのメトリクスと損失値を自動的に保存します。

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Keras でモデルをセットアップするコード

# コールバックを model.fit に渡します。
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

短いビデオ [1 分以内に Keras と Weights & Biases を使い始める](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M) をご覧ください。

より詳細なビデオについては、[Weights & Biases と Keras を統合する](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab_channel=Weights%26Biases) をご覧ください。[Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras_pipeline_with_Weights_and_Biases.ipynb) を確認できます。

{{% alert %}}
スクリプトについては、[example repo](https://github.com/wandb/examples) を参照してください。これには、[Fashion MNIST example](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py) と、それが生成する [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) が含まれています。
{{% /alert %}}

`WandbCallback` クラスは、監視するメトリックの指定、重みと勾配の追跡、training_data と validation_data での予測の記録など、さまざまなログ構成オプションをサポートしています。

詳細については、[`keras.WandbCallback` のリファレンスドキュメント]({{< relref path="/ref/python/integrations/keras/wandbcallback.md" lang="ja" >}}) を確認してください。

`WandbCallback`

* Keras によって収集されたメトリクスから履歴データを自動的に記録します: 損失と `keras_model.compile()` に渡されたもの。
* `monitor` および `mode` 属性で定義されているように、「最良」のトレーニングステップに関連付けられた run の概要メトリクスを設定します。これはデフォルトで、最小の `val_loss` を持つエポックになります。`WandbCallback` はデフォルトで、最良の `epoch` に関連付けられたモデルを保存します。
* オプションで勾配とパラメーターのヒストグラムを記録します。
* オプションで、wandb が可視化するためにトレーニングデータと検証データを保存します。

### `WandbCallback` リファレンス

| 引数 | |
| -------------------------- | ------------------------------------------- |
| `monitor` | (str) 監視するメトリックの名前。デフォルトは `val_loss`。|
| `mode` | (str) {`auto`、`min`、`max`} のいずれか。`min` - モニターが最小化されたときにモデルを保存します `max` - モニターが最大化されたときにモデルを保存します `auto` - モデルを保存するタイミングを推測しようとします (デフォルト)。|
| `save_model` | True - モニターが以前のエポックをすべて上回ったときにモデルを保存します False - モデルを保存しません |
| `save_graph` | (boolean) True の場合、モデルグラフを wandb に保存します (デフォルトは True)。|
| `save_weights_only` | (boolean) True の場合、モデルの重みのみを保存します (`model.save_weights(filepath)`)。それ以外の場合は、完全なモデルを保存します)。|
| `log_weights` | (boolean) True の場合、モデルのレイヤーの重みのヒストグラムを保存します。|
| `log_gradients` | (boolean) True の場合、トレーニング勾配のヒストグラムを記録します |
| `training_data` | (tuple) `model.fit` に渡されるのと同じ形式 `(X,y)`。これは勾配を計算するために必要です。`log_gradients` が `True` の場合は必須です。|
| `validation_data` | (tuple) `model.fit` に渡されるのと同じ形式 `(X,y)`。wandb が可視化するためのデータのセット。このフィールドを設定すると、すべてのエポックで、wandb は少数の予測を行い、後で可視化するために結果を保存します。|
| `generator` | (generator) wandb が可視化するための検証データを返すジェネレーター。このジェネレーターはタプル `(X,y)` を返す必要があります。wandb が特定のデータ例を可視化するには、`validate_data` またはジェネレーターのいずれかを設定する必要があります。|
| `validation_steps` | (`validation_data` がジェネレーターの場合、完全な検証セットに対してジェネレーターを実行するステップ数 (int)。|
| `labels` | (list) wandb でデータを可視化している場合、このラベルのリストは、複数のクラスを持つ分類子を構築している場合に、数値出力を理解可能な文字列に変換します。バイナリ分類子の場合、2 つのラベルのリスト \[`false のラベル`、`true のラベル`] を渡すことができます。`validate_data` と `generator` の両方が false の場合、これは何も行いません。|
| `predictions` | (int) 可視化のために各エポックで行う予測の数。最大は 100 です。|
| `input_type` | (string) 可視化を支援するモデル入力のタイプ。(image、images、segmentation_mask) のいずれかになります。|
| `output_type` | (string) モデル出力のタイプを可視化するのに役立ちます。(image、images、segmentation_mask) のいずれかになります。|
| `log_evaluation` | (boolean) True の場合、各エポックで検証データとモデルの予測を含む Table を保存します。詳細については、`validation_indexes`、`validation_row_processor`、および `output_row_processor` を参照してください。|
| `class_colors` | (\[float, float, float]) 入力または出力がセグメンテーションマスクの場合、各クラスの rgb タプル (範囲 0 ～ 1) を含む配列。|
| `log_batch_frequency` | (integer) None の場合、コールバックはすべてのエポックを記録します。整数に設定すると、コールバックは `log_batch_frequency` バッチごとにトレーニングメトリクスを記録します。|
| `log_best_prefix` | (string) None の場合、追加の概要メトリクスは保存されません。文字列に設定すると、監視対象のメトリックとエポックにプレフィックスを付加し、結果を概要メトリクスとして保存します。|
| `validation_indexes` | (\[wandb.data_types._TableLinkMixin]) 各検証例に関連付けるインデックスキーの順序付きリスト。`log_evaluation` が True で、`validation_indexes` を提供する場合、検証データの Table は作成されません。代わりに、各予測を `TableLinkMixin` で表される行に関連付けます。行キーのリストを取得するには、`Table.get_index() ` を使用します。|
| `validation_row_processor` | (Callable) 検証データに適用する関数。通常はデータを可視化するために使用されます。この関数は、`ndx` (int) と `row` (dict) を受け取ります。モデルに単一の入力がある場合、`row["input"]` には行の入力データが含まれます。それ以外の場合は、入力スロットの名前が含まれます。適合関数が単一のターゲットを受け取る場合、`row["target"]` には行のターゲットデータが含まれます。それ以外の場合は、出力スロットの名前が含まれます。たとえば、入力データが単一の配列である場合、データを画像として可視化するには、プロセッサとして `lambda ndx, row: {"img": wandb.Image(row["input"])}` を指定します。`log_evaluation` が False であるか、`validation_indexes` が存在する場合は無視されます。|
| `output_row_processor` | (Callable) `validation_row_processor` と同じですが、モデルの出力に適用されます。`row["output"]` には、モデル出力の結果が含まれます。|
| `infer_missing_processors` | (Boolean) `validation_row_processor` と `output_row_processor` が欠落している場合に、推論するかどうかを決定します。デフォルトは True です。`labels` を指定すると、W&B は必要に応じて分類タイプのプロセッサを推論しようとします。|
| `log_evaluation_frequency` | (int) 評価結果を記録する頻度を決定します。デフォルトは `0` で、トレーニングの最後にのみ記録します。すべてのエポックで記録するには 1 に、他のすべてのエポックで記録するには 2 に設定します。`log_evaluation` が False の場合は効果がありません。|

## よくある質問

### `Keras` マルチプロセッシングを `wandb` で使用するにはどうすればよいですか?

`use_multiprocessing=True` を設定すると、次のエラーが発生する可能性があります。

```python
Error("You must call wandb.init() before wandb.config.batch_size")
```

これを回避するには:

1. `Sequence` クラスの構築で、`wandb.init(group='...')` を追加します。
2. `main` で、`if __name__ == "__main__":` を使用していることを確認し、残りのスクリプト ロジックをその中に入れます。
```
