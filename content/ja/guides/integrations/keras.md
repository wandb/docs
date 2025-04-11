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

W&B は Keras 用に3つのコールバックを提供しています。`wandb` v0.13.4から利用可能です。レガシーな `WandbCallback` は下にスクロールしてください。

- **`WandbMetricsLogger`** : このコールバックは [Experiment Tracking]({{< relref path="/guides/models/track" lang="ja" >}}) に使用します。トレーニングと検証のメトリクス、システムメトリクスを Weights and Biases にログします。

- **`WandbModelCheckpoint`** : モデルのチェックポイントを Weights and Biases の [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) にログするためにこのコールバックを使用します。

- **`WandbEvalCallback`**: このベースコールバックは、モデルの予測を Weights and Biases の [Tables]({{< relref path="/guides/models/tables/" lang="ja" >}}) にログして、インタラクティブな可視化を行います。

これらの新しいコールバックは以下の特徴を持っています：

* Keras のデザイン哲学に従います。
* すべての機能に対して単一のコールバック (`WandbCallback`) を使用する際の認知負荷を減らします。
* Keras ユーザーがコールバックをサブクラス化してニッチなユースケースをサポートできるように簡単に改修できます。

## `WandbMetricsLogger` を使用して実験を追跡

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}

`WandbMetricsLogger` は、`on_epoch_end` や `on_batch_end` などのコールバックメソッドが引数として取得する Keras の `logs` 辞書を自動的にログします。

これにより次の項目が追跡されます：

* `model.compile` に定義されたトレーニングと検証のメトリクス。
* システム (CPU/GPU/TPU) のメトリクス。
* 学習率（固定値または学習率スケジューラ）。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# 新しい W&B run を初期化
wandb.init(config={"bs": 12})

# WandbMetricsLogger を model.fit に渡す
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbMetricsLogger()]
)
```

### `WandbMetricsLogger` リファレンス

| パラメータ | 説明 |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | (`epoch`, `batch`, または `int`): `epoch` の場合、各エポック終了時にメトリクスをログします。`batch` の場合、各バッチ終了時にメトリクスをログします。`int` の場合、その数のバッチ終了時にメトリクスをログします。デフォルトは `epoch`。 |
| `initial_global_step` | (int): 初期エポックからトレーニングを再開し、かつ学習率スケジューラを使用する場合、学習率を正しくログするためにこの引数を使用します。step_size * initial_step として計算できます。デフォルトは 0。 |

## `WandbModelCheckpoint` を使用してモデルをチェックポイント

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}

`WandbModelCheckpoint` コールバックを使用して、Keras モデル (`SavedModel` 形式) またはモデルの重みを定期的に保存し、モデルのバージョン管理のために W&B アーティファクトとしてアップロードします。

このコールバックは [`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) からサブクラス化されているため、チェックポイントのロジックは親コールバックによって処理されます。

このコールバックが保存するもの：

* モニターに基づいて最高のパフォーマンスを達成したモデル。
* パフォーマンスに関係なく各エポック終了時のモデル。
* エポックまたは一定のトレーニングバッチ数後のモデル。
* モデルの重みのみ、またはモデル全体。
* `SavedModel` 形式または `.h5` 形式いずれかのモデル。

このコールバックは `WandbMetricsLogger` と併用してください。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# 新しい W&B run を初期化
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

| パラメータ | 説明 | 
| ------------------------- |  ---- | 
| `filepath`   | (str): モードファイルを保存するパス。|  
| `monitor`                 | (str): モニターするメトリクスの名前。 |
| `verbose`                 | (int): 冗長モード。0 または 1。モード 0 は静かに動作し、モード 1 はコールバックがアクションをとるときにメッセージを表示します。 |
| `save_best_only`          | (Boolean): `save_best_only=True` の場合、`monitor` と `mode` 属性で定義された要件に基づいて最新のモデルまたはベストとみなされるモデルのみを保存します。 |
| `save_weights_only`       | (Boolean): `True` の場合、モデルの重みのみを保存します。 |
| `mode`                    | (`auto`, `min`, or `max`): `val_acc` の場合は `max` に設定し、`val_loss` の場合は `min` に設定してください。  |
| `save_freq`               | ("epoch" or int): `epoch` を使用する場合、コールバックは各エポック後にモデルを保存します。整数を使用する場合、指定されたバッチ数の終了時にモデルを保存します。`val_acc` や `val_loss` などの検証メトリクスを監視する場合、`save_freq` は "epoch" に設定する必要があります。 |
| `options`                 | (str): `save_weights_only` が真の場合はオプションの `tf.train.CheckpointOptions` オブジェクト、`save_weights_only` が偽の場合はオプションの `tf.saved_model.SaveOptions` オブジェクト。    |
| `initial_value_threshold` | (float): 監視するメトリクスの初期 "ベスト" 値。 |

### N エポック後にチェックポイントをログ

デフォルト (`save_freq="epoch"`) では、コールバックは各エポック後にアーティファクトとしてチェックポイントを作成し、アップロードします。特定のバッチ数後にチェックポイントを作成するには、`save_freq` を整数に設定します。`N` エポック後にチェックポイントを作成するには、`train` データローダーの基数を計算し、それを `save_freq` に渡します。

```python
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### TPU アーキテクチャーで効率的にチェックポイントをログ

TPU 上でチェックポイントを作成する際に、`UnimplementedError: File system scheme '[local]' not implemented` エラーメッセージが発生することがあります。これは、モデルディレクトリー (`filepath`) がクラウドストレージバケットパス (`gs://bucket-name/...`) を使用しなければならないためであり、このバケットは TPU サーバーからアクセス可能でなければなりません。ただし、ローカルパスを使用してチェックポイントを行い、それを Artifacts としてアップロードすることは可能です。

```python
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## モデル予測を `WandbEvalCallback` で可視化

{{< cta-button colabLink="https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}

`WandbEvalCallback` は、モデル予測のための Keras コールバックを主に構築するための抽象基底クラスであり、副次的にデータセットの可視化にも使われます。

この抽象コールバックは、データセットやタスクに対してはアグノスティックです。これを使用するには、このベース `WandbEvalCallback` コールバッククラスを継承し、`add_ground_truth` と `add_model_prediction` メソッドを実装します。

`WandbEvalCallback` は、以下のメソッドを提供するユーティリティクラスです：

* データと予測の `wandb.Table` インスタンスを作成します。
* データと予測のテーブルを `wandb.Artifact` としてログします。
* `on_train_begin` 時にデータテーブルをログします。
* `on_epoch_end` 時に予測テーブルをログします。

以下の例では、画像分類タスクのために `WandbClfEvalCallback` を使用しています。この例のコールバックは検証データ (`data_table`) を W&B にログし、推論を行い、各エポック終了時に予測 (`pred_table`) を W&B にログします。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback


# モデル予測可視化用コールバックを実装
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

# 新しい W&B run を初期化
wandb.init(config={"hyper": "parameter"})

# コールバックを Model.fit に追加
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
W&B の [Artifact ページ]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ja" >}}) には、デフォルトでテーブルログが含まれており、**Workspace** ページには含まれていません。
{{% /alert %}}

### `WandbEvalCallback` リファレンス

| パラメータ            | 説明                      |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (list) `data_table` の列名のリスト               |
| `pred_table_columns` | (list) `pred_table` の列名のリスト               |

### メモリ使用量の詳細

`data_table` は `on_train_begin` メソッドが呼び出されたときに W&B にログされます。一度 W&B アーティファクトとしてアップロードされると、`data_table_ref` クラス変数を使用してこのテーブルにアクセスすることができます。`data_table_ref` は 2D リストで、`self.data_table_ref[idx][n]` のようにインデックスを付けてアクセスできます。この例では、`idx` は行番号で、`n` は列番号です。

### コールバックのカスタマイズ

`on_train_begin` や `on_epoch_end` メソッドをオーバーライドして、より細かい制御を行うことができます。`N` バッチ後にサンプルをログしたい場合、`on_train_batch_end` メソッドを実装することができます。

{{% alert %}}
💡 `WandbEvalCallback` を継承してモデル予測可視化のコールバックを実装している場合、何か明確にすべき点や修正が必要な場合は、問題を報告してお知らせください。[issue](https://github.com/wandb/wandb/issues) を開いてください。
{{% /alert %}}

## `WandbCallback` [レガシー]

`WandbCallback` クラスを使用して、`model.fit` で追跡されるすべてのメトリクスと損失値を自動的に保存します。

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Keras でモデルをセットアップするためのコード

# コールバックを model.fit に渡す
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

短いビデオ [Get Started with Keras and Weights & Biases in Less Than a Minute](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M) をご覧ください。

より詳細なビデオは [Integrate Weights & Biases with Keras](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab_channel=Weights%26Biases) をご覧ください。[Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras_pipeline_with_Weights_and_Biases.ipynb) を確認できます。

{{% alert %}}
スクリプトを含む私たちの [example repo](https://github.com/wandb/examples) をご覧ください。ここには [Fashion MNISTの例](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py) とそれが生成する [W&B ダッシュボード](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) があります。
{{% /alert %}}

`WandbCallback` クラスは、広範なロギング設定オプションをサポートしています：監視するメトリクスの指定、重みや勾配の追跡、トレーニングデータと検証データ上の予測のログなど。

`keras.WandbCallback` の参考文献のドキュメントも確認してください。より詳細な情報があります。

`WandbCallback` 

* Keras によって収集された任意のメトリクスの履歴データを自動的にログします：`keras_model.compile()` に渡された損失とその他の項目。
* `monitor` と `mode` 属性によって定義された "最良" のトレーニングステップに関連付けられたサマリーメトリクスを設定します。これはデフォルトでは最小の `val_loss` を持つエポックとなります。`WandbCallback` はデフォルトで最も良い `epoch` に関連付けられたモデルを保存します。
* 勾配とパラメータのヒストグラムをオプションでログします。
* オプションで wandb に視覚化するためのトレーニングおよび検証データを保存します。

### `WandbCallback` リファレンス

| 引数                       |                                    |
| -------------------------- | ------------------------------------------- |
| `monitor`                  | (str) monitor するメトリックの名前。デフォルトは `val_loss`。                                                                   |
| `mode`                     | (str) {`auto`, `min`, `max`} のいずれか。`min` - モニターが最小化されるときにモデルを保存 `max` - モニターが最大化されるときにモデルを保存 `auto` - モデル保存のタイミングを推測（デフォルト）。                                                                                                                                                |
| `save_model`               | True - monitor が過去のすべてのエポックを上回った場合にモデルを保存 False - モデルを保存しない                                       |
| `save_graph`               | (boolean) True の場合、wandb にモデルグラフを保存します（デフォルトは True）。                                                 |
| `save_weights_only`        | (boolean) True の場合、モデルの重みのみを保存します（`model.save_weights(filepath)`）。そうでなければ、完全なモデルを保存します。   |
| `log_weights`              | (boolean) True の場合、モデルのレイヤーの重みのヒストグラムを保存します。                                                 |
| `log_gradients`            | (boolean) True の場合、トレーニング勾配のヒストグラムをログします                                                     |
| `training_data`            | (tuple) `model.fit` に渡される `(X,y)` と同じ形式。勾配を計算するために必要で、`log_gradients` が True の場合必須です。       |
| `validation_data`          | (tuple) `model.fit` に渡される `(X,y)` と同じ形式。Wandb が視覚化するためのデータセット。フィールドを設定すると、各エポックで wandb が少数の予測を行い、視覚化のための結果を保存します。          |
| `generator`                | (generator) wandb が視覚化するための検証データを返すジェネレータ。このジェネレータはタプル `(X,y)` を返すべきです。`validate_data` またはジェネレータのいずれかをセットすることで、wandb は特定のデータ例を視覚化できます。     |
| `validation_steps`         | (int) `validation_data` がジェネレータの場合、完全な検証セットのためにジェネレータを実行するステップ数。       |
| `labels`                   | (list) wandb でデータを視覚化する場合、複数クラスの分類器を構築する際の数値出力を理解しやすい文字列に変換するラベルのリスト。バイナリ分類器の場合、2つのラベル [`label for false`, `label for true`] を渡すことができます。`validate_data` と `generator` の両方がfalseの場合は何も行いません。    |
| `predictions`              | (int) 各エポックの視覚化のために行う予測の数。最大は 100 です。 |
| `input_type`               | (string) 視覚化を助けるためのモデル入力の型。`image`、`images`、`segmentation_mask` のいずれか。  |
| `output_type`              | (string) 視覚化を助けるためのモデル出力の型。`image`、`images`、`segmentation_mask` のいずれか。  |
| `log_evaluation`           | (boolean) True の場合、各エポックで検証データとモデルの予測を含むテーブルを保存します。詳細は `validation_indexes`、`validation_row_processor`、`output_row_processor` を参照してください。     |
| `class_colors`             | (\[float, float, float]) 入力または出力がセグメンテーションマスクの場合、各クラスのための RGB タプル（範囲 0-1）を含む配列。                  |
| `log_batch_frequency`      | (integer) None の場合、コールバックは各エポックをログします。整数を設定する場合、コールバックは `log_batch_frequency` バッチごとにトレーニングメトリクスをログします。          |
| `log_best_prefix`          | (string) None の場合、追加のサマリーメトリクスを保存しません。文字列が設定されている場合、プレフィックスとともに監視されたメトリクスとエポックを毎回保存し、サマリーメトリクスとして保存します。   |
| `validation_indexes`       | (\[wandb.data_types._TableLinkMixin]) 各検証例に関連付けるインデックスキーの順序付きリスト。`log_evaluation` が True で `validation_indexes` を提供する場合、検証データのテーブルを作成しません。その代わり、各予測を `TableLinkMixin` で表される行に関連付けます。行のキーを取得するには、`Table.get_index()` を使用します。        |
| `validation_row_processor` | (Callable) 検証データに適用される関数で、一般にデータを視覚化するのに使用します。関数には `ndx` (int) と `row` (dict) が渡されます。モデルに単一の入力がある場合、`row["input"]` はその行の入力データを含みます。そうでない場合、入力スロットの名前を含みます。`fit` フィット関数が単一のターゲットを取り込む場合、`row["target"]` はその行のターゲットデータを含みます。異なるアウトプットスロットの名前を含んでいます。たとえば、入力データが単一の配列で、そのデータをImageとして視覚化するためには、`lambda ndx, row: {"img": wandb.Image(row["input"])}` をプロセッサとして提供します。`log_evaluation` がFalseの場合や `validation_indexes` が存在する場合は無視されます。 |
| `output_row_processor`     | (Callable) `validation_row_processor` と同様だが、モデルの出力に適用されます。`row["output"]` はモデルの出力結果を含みます。  |
| `infer_missing_processors` | (Boolean) `validation_row_processor` と `output_row_processor` が欠落している場合にそれを推測するかどうかを決定します。デフォルトでは True。`labels` を提供すると、W&B は適切な場合に分類タイプのプロセッサを推測しようとします。  |
| `log_evaluation_frequency` | (int) 評価結果の記録頻度を決定します。デフォルトは `0` で、トレーニングの終了時にのみログします。1に設定すると各エポックごとにログします。2ならば隔エポックでログします。`log_evaluation` が False のときには効果がありません。    |

## よくある質問

### `Keras` のマルチプロセッシングを `wandb` でどのように使用しますか？

`use_multiprocessing=True` を設定したときにこのエラーが発生する場合があります：

```python
Error("You must call wandb.init() before wandb.config.batch_size")
```

それを回避する方法：

1. `Sequence` クラスの構築時に、これを追加：`wandb.init(group='...')`。
2. `main` で、`if __name__ == "__main__":` を使用して、スクリプトロジックの残りをその中に置くようにしてください。