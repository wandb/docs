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

W&B には Keras 用の 3 つのコールバックがあります。`wandb` バージョン 0.13.4 から利用可能です。レガシーの `WandbCallback` については下の方をご覧ください。

- **`WandbMetricsLogger`** : [実験管理]({{< relref path="/guides/models/track" lang="ja" >}}) 用にこのコールバックを使用してください。トレーニングや検証のメトリクスと、システムのメトリクスを Weights & Biases にログします。

- **`WandbModelCheckpoint`** : モデルのチェックポイントを Weights & Biases の [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) にログするために、このコールバックを使用してください。

- **`WandbEvalCallback`**: 基本的なコールバックで、モデルの予測を Weights & Biases [Tables]({{< relref path="/guides/core/tables/" lang="ja" >}}) にログして、インタラクティブな可視化を可能にします。

これらの新しいコールバックは以下の特長があります：

* Keras のデザイン哲学に従っています。
* すべてを単一のコールバック (`WandbCallback`) で処理する際の認知負荷を軽減します。
* Keras ユーザーがニッチなユースケースに対応するためにサブクラス化してコールバックを修正しやすくします。

## `WandbMetricsLogger` で実験をトラックする

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}

`WandbMetricsLogger` は、コールバックメソッド `on_epoch_end`、`on_batch_end` などが引数として受け取る Keras の `logs` 辞書を自動でログします。

トラックできるもの：

* `model.compile` で定義されたトレーニングと検証メトリクス。
* システム（CPU/GPU/TPU）メトリクス。
* 学習率（固定値または学習率スケジューラのどちらでも）。

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
| `log_freq`            | (`epoch`, `batch`, または `int`): `epoch`の場合、各エポックの終了時にメトリクスをログ。`batch`の場合、各バッチの終了時にメトリクスをログ。`int`のときは、その多くのバッチの終了時にメトリクスをログ。デフォルトは `epoch`。 |
| `initial_global_step` | (int): トレーニングの再開時、および学習率スケジューラが使用されるときに、正しく学習率をログするためにこの引数を使用します。これは step_size * initial_step で計算できます。デフォルトは 0。 |

## モデルを `WandbModelCheckpoint` でチェックポイントする

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}

`WandbModelCheckpoint` コールバックを使用して、Keras モデル (`SavedModel` 形式) またはモデルの重みを定期的に保存し、`wandb.Artifact` として W&B にアップロードしてモデルのバージョン管理を行います。

このコールバックは、[`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) からサブクラス化されており、チェックポイントのロジックは親コールバックによって処理されます。

このコールバックが保存するもの：

* モニターに基づいて最高の性能を達成したモデル。
* パフォーマンスに関係なく、各エポックの終了時のモデル。
* エポックの終了時または固定数のトレーニングバッチ後のモデル。
* モデルの重みのみまたは全体のモデル。
* モデルを `SavedModel` 形式か `.h5` 形式のいずれかで。

このコールバックを `WandbMetricsLogger` と併用してください。

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
| `monitor`                 | (str): モニターするメトリクスの名前。         |
| `verbose`                 | (int): 冗長モード、0 または 1。モード 0 はサイレント、モード 1 はアクションを実行する際のメッセージを表示します。   |
| `save_best_only`          | (Boolean): `save_best_only=True` の場合、定義された `monitor` と `mode` 属性に従って最新モデルまたは最良と見なされるモデルのみを保存します。 |
| `save_weights_only`       | (Boolean): True の場合、モデルの重みのみ保存します。                                            |
| `mode`                    | (`auto`, `min`, または `max`): `val_acc` 用には `max` に設定し、`val_loss` 用には `min` に設定します。  |                     |
| `save_freq`               | ("epoch" または int): ‘epoch’ 使用時、各エポック後にモデルを保存します。整数使用時、このバッチ数の終了時にモデルを保存します。`val_acc` または `val_loss` などの検証メトリクスをモニタリングする場合は、これらのメトリクスがエポックの終了時にのみ利用可能であるため、`save_freq` を "epoch" に設定しなければなりません。 |
| `options`                 | (str): `save_weights_only` が true の場合はオプションの `tf.train.CheckpointOptions` オブジェクト、`save_weights_only` が false の場合はオプションの `tf.saved_model.SaveOptions` オブジェクトです。    |
| `initial_value_threshold` | (float): モニタリングされるメトリクスのフローティングポイント初期 "best" 値。       |

### N アーツごとのチェックポイントログ

デフォルトでは (`save_freq="epoch"`)、コールバックは各エポック後にチェックポイントを作成し、アーティファクトとしてアップロードします。特定のバッチ数後にチェックポイントを作成するには、`save_freq` を整数に設定します。`N` エポック後にチェックポイントを作成するには、`train` dataloader のカーディナリティを計算し、それを `save_freq` に渡します：

```python
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### TPU アーキテクチャで効率的にチェックポイントをログ

TPU でチェックポイントを取ると、`UnimplementedError: File system scheme '[local]' not implemented` というエラーメッセージが表示されることがあります。これは、モデルディレクトリ (`filepath`) がクラウドストレージバケットパス (`gs://bucket-name/...`) を使用する必要があり、このバケットが TPU サーバーからアクセス可能である必要があるためです。ただし、チェックポイントにはローカルパスを使用し、それが後で Artifacts としてアップロードされることも可能です。

```python
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## `WandbEvalCallback` を使用してモデル予測を可視化

{{< cta-button colabLink="https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}

`WandbEvalCallback` は、主にモデル予測、二次的にデータセット可視化のために Keras コールバックを構築するための抽象基底クラスです。

この抽象コールバックは、データセットとタスクに関して中立的です。これを使用するには、この基盤となる `WandbEvalCallback` コールバッククラスを継承し、`add_ground_truth` と `add_model_prediction` メソッドを実装します。

`WandbEvalCallback` は以下のメソッドを提供するユーティリティクラスです：

* データおよび予測 `wandb.Table` インスタンスを作成します。
* データおよび予測テーブルを `wandb.Artifact` としてログします。
* データテーブルを `on_train_begin` でログします。
* 予測テーブルを `on_epoch_end` でログします。

以下の例では、画像分類タスク用に `WandbClfEvalCallback` を使用しています。 これは、検証データ（`data_table`）を W&B にログし、推論を行い、各エポック終了時に予測（`pred_table`）を W&B にログします。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback


# モデル予測可視化コールバックを実装
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

# モデルにコールバックを追加
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
W&B の [Artifact ページ]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ja" >}}) には、デフォルトでテーブルログが含まれていますが、**Workspace** ページには含まれていません。
{{% /alert %}}

### `WandbEvalCallback` リファレンス

| パラメータ            | 説明                                      |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (list) `data_table` の列名のリスト |
| `pred_table_columns` | (list) `pred_table` の列名のリスト |

### メモリフットプリントの詳細

`on_train_begin` メソッドが呼ばれたときに、`data_table` を W&B にログします。一旦 W&B アーティファクトとしてアップロードすると、`data_table_ref` クラス変数を使用してこのテーブルにアクセスできます。`data_table_ref` は `self.data_table_ref[idx][n]` のようにインデックス化可能な 2D リストです。`idx` は行番号であり、`n` は列番号です。以下の例でその使用を確認しましょう。

### コールバックのカスタマイズ

`on_train_begin` や `on_epoch_end` メソッドをオーバーライドして、よりきめ細かい制御を入れることができます。`N` バッチ後にサンプルをログしたい場合、`on_train_batch_end` メソッドを実装することができます。

{{% alert %}}
💡 `WandbEvalCallback` を継承してモデル予測可視化用のコールバックを実装しており、何か明確化または修正が必要な場合は、[issue](https://github.com/wandb/wandb/issues) を開いてお知らせください。
{{% /alert %}}

## `WandbCallback` [レガシー]

W&B ライブラリの [`WandbCallback`]({{< relref path="/ref/python/integrations/keras/wandbcallback" lang="ja" >}}) クラスを使用して、自動的にすべてのメトリクスと `model.fit` で追跡される損失値を保存します。

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Keras でモデルを設定するためのコード

# コールバックを model.fit に渡す
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

短いビデオ [Get Started with Keras and Weights & Biases in Less Than a Minute](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M) をご覧ください。

より詳しいビデオは [Integrate Weights & Biases with Keras](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab_channel=Weights%26Biases) をご覧ください。[Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras_pipeline_with_Weights_and_Biases.ipynb) を確認することもできます。

{{% alert %}}
スクリプトを含む [例リポジトリ](https://github.com/wandb/examples) を参照し、[Fashion MNIST の例](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py) と、それが生成する [W&B ダッシュボード](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) を確認してください。
{{% /alert %}}

`WandbCallback` クラスは、多様なログ設定オプションをサポートしています：モニターするメトリクスの指定、重みと勾配の追跡、トレーニングデータと検証データの予測のログなど。

`keras.WandbCallback` の [リファレンスドキュメント]({{< relref path="/ref/python/integrations/keras/wandbcallback.md" lang="ja" >}}) をチェックして詳細を確認してください。

`WandbCallback` 

* Keras によって収集された任意のメトリクスから履歴データを自動的にログします：損失と `keras_model.compile()` に渡されたすべて。
* `monitor` と `mode` 属性によって定義された "best" なトレーニングステップに関連付けられたサマリーメトリクスを設定します。デフォルトでは、最小の `val_loss` を持つエポックです。`WandbCallback` はデフォルトで最良の `epoch` に関連付けられたモデルを保存します。
* オプションで、勾配およびパラメータヒストグラムをログします。
* オプションで、wandb が可視化するためにトレーニングおよび検証データを保存します。

### `WandbCallback` リファレンス

| 引数                   |                                    |
| -------------------------- | ------------------------------------------- |
| `monitor`                  | (str) 監視するメトリクスの名前。デフォルトは `val_loss`。 |
| `mode`                     | (str) {`auto`, `min`, `max`} のいずれか。`min` - monitor が最小化されたときにモデルを保存します。`max` - monitor が最大化されたときにモデルを保存します。`auto` - モデルの保存タイミングを推測します（デフォルト）。 |
| `save_model`               | True - すべての先行するエポックより良いと判断したときにモデルを保存します。False - モデルを保存しません。 |
| `save_graph`               | (boolean) True のとき、モデルのグラフを wandb に保存します（デフォルトは True）。 |
| `save_weights_only`        | (boolean) True のとき、モデルの重みのみを保存します（`model.save_weights(filepath)）。それ以外の場合、モデル全体を保存します。 |
| `log_weights`              | (boolean) True のとき、モデルのレイヤーの重みのヒストグラムを保存します。 |
| `log_gradients`            | (boolean) True のとき、トレーニング勾配のヒストグラムをログします。 |
| `training_data`            | (tuple) `model.fit` に渡された `(X,y)`と同じ形式。この情報は勾配の計算に必要であり、`log_gradients` が `True` のときに必要です。 |
| `validation_data`          | (tuple) `model.fit` に渡された形式 `(X,y)” と同じ。wandb が可視化するためのデータセット。毎エポック、少数の予測を行い、後で視覚化のために結果を保存します。 |
| `generator`                | (generator) wandb が可視化するためのデータを返すジェネレーター。このジェネレーターは `(X,y)` のタプルを返す必要があります。wandb が特定のデータ例を可視化するために、`validate_data` または generator をセットする必要があります。 |
| `validation_steps`         | (int) `validation_data` がジェネレーターである場合、フルバリデーションセットのためにジェネレーターを実行するステップ数。 |
| `labels`                   | (list) wandb でデータを可視化する場合、このラベルリストは数値出力を理解可能な文字列に変換します。多クラス分類を構築している場合は、この二重ラベルのリストを渡すことができます。{`false` のためのラベル, `true` のためのラベル} 。`validate_data` と `generator` がどちらも false の場合、何もしません。 |
| `predictions`              | (int) 各エポックごとにビジュアリゼーション用に行う予測の数、最大は 100 です。 |
| `input_type`               | (string) ビジュアリゼーションを助けるためのモデル入力のタイプ。以下のいずれか: (`image`, `images`, `segmentation_mask`)。 |
| `output_type`              | (string) ビジュアリゼーションを助けるためのモデル出力のタイプ。以下のいずれか: (`image`, `images`, `segmentation_mask`)。 |
| `log_evaluation`           | (boolean) True の場合、各エポックで評価データとモデルの予測を含むテーブルを保存します。詳細については `validation_indexes`、`validation_row_processor`、および `output_row_processor` を参照してください。 |
| `class_colors`             | (\[float, float, float]) 入出力がセグメンテーションマスクの場合、各クラスの rgb タプル（範囲 0-1）を含む配列として指定します。 |
| `log_batch_frequency`      | (integer) None の場合、コールバックは各エポックをログします。整数を指定した場合、`log_batch_frequency` バッチごとにトレーニングメトリクスをログします。 |
| `log_best_prefix`          | (string) None の場合、追加のサマリーメトリクスを保存しません。文字列を設定した場合、監視されたメトリクスとエポックのプレフィックスを設定し、結果をサマリーメトリクスとして保存します。 |
| `validation_indexes`       | (\[wandb.data_types._TableLinkMixin]) 検証例ごとに関連付けるインデックスキーの順序付きリスト。`log_evaluation` が True の場合、`validation_indexes` を提供すると、検証データのテーブルを作成しません。代わりに、各予測を `TableLinkMixin` が表す行に関連付けます。行キーのリストは `Table.get_index()` を使って取得できます。 |
| `validation_row_processor` | (Callable) 検証データに適用する関数、よくビジュアリゼーションのために使われます。この関数は `ndx` (int) と `row` (dict) を受け取ります。モデルが単一入力を持つ場合、`row["input"]` は行の入力データを含みます。そうでない場合、入力スロットの名前を含みます。フィット関数が単一ターゲットを取る場合、`row["target"]` は行のターゲットデータを含みます。それ以外の場合、出力スロットの名前を含みます。例えば、入力データが単一配列である場合、データを画像として視覚化するには `lambda ndx, row: {"img": wandb.Image(row["input"])}` をプロセッサとして提供します。`log_evaluation` が False の場合や `validation_indexes` が存在する場合は無視されます。 |
| `output_row_processor`     | (Callable) `validation_row_processor` と同様、モデルの出力に適用される関数。同じように `row["output"]` はモデル出力の結果を含みます。 |
| `infer_missing_processors` | (Boolean) `validation_row_processor` と `output_row_processor` が欠けている場合に、それを推論するかどうかを決定します。デフォルトは True。`labels` を指定する場合、W&B は適切な場所で分類型プロセッサを推測しようとします。 |
| `log_evaluation_frequency` | (int) 評価結果をログする頻度を決定します。デフォルトは `0` で、トレーニング終了時にのみログします。1 を設定すると毎エポック、2 を設定すると隔エポック、などのように設定します。`log_evaluation` が False の場合は効果がありません。 |

## よくある質問

### `wandb` を使って `Keras` のマルチプロセッシングを使用する方法は？

`use_multiprocessing=True` を設定した際に、このエラーが発生するかもしれません：

```python
Error("You must call wandb.init() before wandb.config.batch_size")
```

これを回避するには：

1. `Sequence` クラスのコンストラクションで次を追加します：`wandb.init(group='...')`。
2. `main` 内で、`if __name__ == "__main__":` を使用して、スクリプトの残りのロジックをその中に入れます。