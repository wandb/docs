---
displayed_sidebar: default
---
# Keras

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/intro-keras)

## Weights & Biases Kerasコールバック

`wandb` v0.13.4から、KerasおよびTensorFlowユーザー向けに3つの新しいコールバックを追加しました。従来の`WandbCallback`については、下にスクロールしてください。

### コールバック

**`WandbMetricsLogger`** : このコールバックは、[実験トラッキング](https://docs.wandb.ai/guides/track) に使用します。トレーニングおよび検証メトリクスとシステムメトリクスをWeights and Biasesに記録します。

**`WandbModelCheckpoint`** : このコールバックを使用して、モデルのチェックポイントをWeights and Biasesの[アーティファクト](https://docs.wandb.ai/guides/data-and-model-versioning)に記録します。

**`WandbEvalCallback`**: このベースコールバックは、モデルの予測をWeights and Biasesの[テーブル](https://docs.wandb.ai/guides/data-vis)に記録し、インタラクティブな可視化を行います。

これらの新しいコールバックには、

* Kerasのデザイン哲学に従っています
* 単一のコールバック（`WandbCallback`）ですべてを行う際の認知負荷を軽減しています。
* Kerasユーザーがサブクラス化してコールバックを変更し、ニッチなユースケースに対応できるようにしています。

### `WandbMetricsLogger`を使った実験トラッキング

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/wandb/examples/blob/master/colabs/keras/Use\_WandbMetricLogger\_in\_your\_Keras\_workflow.ipynb)

`WandbMetricsLogger`は、`on_epoch_end`、`on_batch_end`などのコールバックメソッドが引数として取るKerasの`logs`ディクショナリを自動的に記録します。

これにより、以下のことが提供されます。

* `model.compile`で定義されたトレーニングと検証メトリクス
* システム（CPU/GPU/TPU）メトリクス
* 学習率（固定値または学習率スケジューラの両方）
```python
import wandb
from wandb.keras import WandbMetricsLogger

# 新しいW&B runを初期化
wandb.init(config={"bs": 12})

# WandbMetricsLoggerをmodel.fitに渡す
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[WandbMetricsLogger()]
)
```

### `WandbMetricsLogger` リファレンス

| パラメータ | 説明 | 
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | ("epoch", "batch", または int): "epoch" の場合、各エポックの終わりにメトリクスを記録します。 "batch" の場合、各バッチの終わりにメトリクスを記録します。int の場合、その数のバッチの終わりにメトリクスを記録します。デフォルトは "epoch"。|
| `initial_global_step` | (int): 初期エポックからトレーニングを再開し、学習率スケジューラーが使用される場合、学習率を正しく記録するためにこの引数を使用します。これは、step_size * initial_step として計算できます。デフォルトは0。 |

## `WandbModelCheckpoint` を使用したモデルのチェックポイント作成

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/wandb/examples/blob/master/colabs/keras/Use\_WandbModelCheckpoint\_in\_your\_Keras\_workflow.ipynb)

`WandbModelCheckpoint` コールバックを使用して、Kerasモデル（`SavedModel`形式）またはモデルの重みを定期的に保存し、それらをW&Bの`wandb.Artifact`としてアップロードしてモデルのバージョン管理を行います。
このコールバックは、[`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)からサブクラス化されているため、チェックポイントのロジックは親コールバックによって処理されます。

このコールバックは以下の機能を提供します:

* "monitor"に基づいた "最高のパフォーマンス" を達成したモデルを保存します。
* パフォーマンスに関係なく、すべてのエポックの終わりにモデルを保存します。
* エポックの終わりまたはトレーニングバッチの一定数後にモデルを保存します。
* モデルの重みのみを保存するか、モデル全体を保存します。
* SavedModel形式または `.h5` 形式でモデルを保存します。

このコールバックは、`WandbMetricsLogger`と併用して使用する必要があります。

```python
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

# 新しいW&B runを初期化
wandb.init(config={"bs": 12})

# WandbModelCheckpointをmodel.fitに渡す
model.fit(
  X_train,
  y_train,
  validation_data=(X_test, y_test),
  callbacks=[
    WandbMetricsLogger(),
    WandbModelCheckpoint("models"),
  ]
)
```
**`WandbModelCheckpoint` リファレンス**

| パラメータ | 説明 |
| ------------------------- | ---- |
| `filepath` | (str): モデルファイルを保存するパス。|
| `monitor` | (str): 監視するメトリクスの名前。|
| `verbose` | (int): 冗長モード、0 または 1。モード 0 は無音で、モード 1 はコールバックがアクションを実行するときにメッセージを表示します。|
| `save_best_only` | (bool): `save_best_only=True` の場合、モデルが「最良」と見なされる場合にのみ保存し、監視対象の数量（`monitor`）による最新の最良モデルは上書きされません。|
| `save_weights_only` | (bool): True の場合、モデルの重みのみが保存されます。|
| `mode` | ("auto", "min", または "max"): val\_acc の場合は「max」、val\_loss の場合は「min」など。|
| `save_freq` | ("epoch" または int): 「epoch」を使用すると、コールバックは各エポックの後にモデルを保存します。整数を使用すると、コールバックはこのバッチ数の終わりにモデルを保存します。ただし、`val_acc` や `val_loss` などの検証メトリクスを監視する場合、これらのメトリクスはエポックの終わりにのみ利用できるため、`save_freq` を "epoch" に設定する必要があります。|
| `options` | (str): `save_weights_only` が true の場合はオプションの `tf.train.CheckpointOptions` オブジェクト、false の場合はオプションの `tf.saved_model.SaveOptions` オブジェクト。|
| `initial_value_threshold` | (float): 監視対象のメトリックの最初の "ベスト" 値の浮動小数点。|

## `WandbEvalCallback` を使用したモデル予測の可視化

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb)

`WandbEvalCallback`は、主にモデル予測と、二次的にデータセットの可視化を目的としたKerasコールバックを構築するための抽象基本クラスです。

この抽象コールバックは、データセットやタスクに関してはアグノスティックです。これを使用するには、この基本的な `WandbEvalCallback` コールバッククラスを継承し、`add_ground_truth` および `add_model_prediction` メソッドを実装してください。

`WandbEvalCallback` は、以下の機能を提供するユーティリティクラスです。

* データと予測の `wandb.Table` インスタンスを作成する
* データと予測のテーブルを `wandb.Artifact` としてログに記録する
* `on_train_begin` でデータテーブルをログに記録する
* `on_epoch_end` で予測テーブルをログに記録する
例えば、画像分類タスクのために以下の`WandbClfEvalCallback`を実装しました。この例のコールバックは:

* W&Bに検証データ（`data_table`）をログする
* 各エポックの終わりに推論を実行し、予測（`pred_table`）をW&Bにログする

```python
import wandb
from wandb.keras import WandbMetricsLogger, WandbEvalCallback

# モデル予測の可視化コールバックを実装
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

# W&B runを新規作成
wandb.init(config={"hyper": "parameter"})

# Model.fitにコールバックを追加
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
    ]
)
```

:::info
💡 テーブルはデフォルトでW&Bの[アーティファクトページ](https://docs.wandb.ai/ref/app/pages/project-page#artifacts-tab)にログされ、[ワークスペース](https://docs.wandb.ai/ref/app/pages/workspaces)ページにはログされません。
:::

### メモリ使用量がどのように削減されるか？

`on_train_begin`メソッドが呼び出されたときに、`data_table`をW&Bにログします。W&Bのアーティファクトとしてアップロードされると、このテーブルにアクセスできる`data_table_ref`クラス変数の参照が得られます。`data_table_ref`は2次元リストであり、`self.data_table_ref[idx][n]`のようにインデックス化できます。ここで、`idx`は行番号、`n`は列番号です。以下の例で使用方法を確認しましょう。
### コールバックをさらにカスタマイズ

より細かい制御を行うために、`on_train_begin`や`on_epoch_end`メソッドをオーバーライドすることができます。`N`バッチ後にサンプルをログに記録したい場合は、`on_train_batch_end`メソッドを実装できます。

:::info
💡 `WandbEvalCallback`を継承してモデル予測の可視化のためのコールバックを実装していて、何か明確にしたり修正したりする必要がある場合は、[issue](https://github.com/wandb/wandb/issues)を開いてお知らせください。
:::

### `WandbEvalCallback` リファレンス

| パラメータ             | 説明                                               |
| ------------------- | ------------------------------------------------ |
| `data_table_columns` | (list) `data_table` の列名のリスト                        |
| `pred_table_columns` | (list) `pred_table` の列名のリスト                        |

## WandbCallback [Legacy]

W&Bライブラリ [`WandbCallback`](https://docs.wandb.ai/ref/python/integrations/keras/wandbcallback) クラスを使用して、`model.fit` でトラッキングされたすべてのメトリクスと損失値を自動的に保存します。

```python
import wandb
from wandb.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Kerasでモデルを設定するコード

# コールバックをmodel.fitに渡す
model.fit(
  X_train,
  y_train,
  validation_data=(X_test, y_test),
  callbacks=[WandbCallback()]
)
```
## 使用例

W&BとKerasを初めて組み合わせる場合は、この1分間のステップバイステップのビデオを参照してください: [KerasとWeights & Biasesを1分以内に始める](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M)

より詳細なビデオについては、[Weights & BiasesとKerasの統合](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab\_channel=Weights%26Biases)を参照してください。使用されているノートブックの例はこちらから見つけることができます: [Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras\_pipeline\_with\_Weights\_and\_Biases.ipynb).

:::info
上記の動画でのW&BとKerasの統合例を[Colabノートブック](http://wandb.me/keras-colab)で試してみてください。または、[Fashion MNISTの例](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py)やそれが生成する[W&Bダッシュボード](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs)などのスクリプトが含まれている[example repo](https://github.com/wandb/examples)を参照してください。
:::

## `WandbCallback`の設定

`WandbCallback`クラスは、多様なロギング設定オプションをサポートしています。監視する指標の指定、重みと勾配の追跡、トレーニング\_データや検証\_データに対する予測のログ記録などが含まれます。

詳細については、[`keras.WandbCallback`のリファレンスドキュメント](../../ref/python/integrations/keras/wandbcallback.md)をご覧ください。

`WandbCallback`

* Kerasで収集されたメトリクスから自動的に履歴データをログする：`loss`や`keras_model.compile()`に渡されたものが対象です。
* 「最適」なトレーニングステップに関連付けられたrunのサマリーメトリクスを設定します。「最適」は`monitor`と`mode`属性で定義され、デフォルトでは最小の`val_loss`を持つエポックになります。`WandbCallback`は、デフォルトで最適な`epoch`に関連付けられたモデルを保存します。
* オプションで勾配とパラメータのヒストグラムをログすることができます。
* オプションでWandbが可視化するためのトレーニングデータと検証データを保存することができます。

### `WandbCallback`リファレンス

 | 引数                         |                                    |
| -------------------------- | ------------------------------------------- |
| `monitor`                  | (str) 監視するメトリックの名前。デフォルトは `val_loss`。                                                                   |
| `mode`                     | (str) {`auto`、`min`、`max`} のいずれか。`min` - 監視対象が最小になった時にモデルを保存 `max` - 監視対象が最大になった時にモデルを保存 `auto` - モデルをいつ保存するか自動的に推測（デフォルト）。                                                                                                                                               |
| `save_model`               | True - 監視がすべての前のエポックを上回った時にモデルを保存する False - モデルを保存しない                                       |
| `save_graph`               | (boolean) True の場合、モデルのグラフを wandb に保存します（デフォルトは True）。                                                           |

| `save_weights_only`        | (boolean) Trueの場合、モデルの重みだけが保存されます（`model.save_weights(filepath)`）、それ以外の場合は、モデル全体が保存されます（`model.save(filepath)`）。   |
| `log_weights`              | (boolean) Trueの場合、モデルのレイヤーの重みのヒストグラムを保存します。                                                |
| `log_gradients`            | (boolean) Trueの場合、トレーニング勾配のヒストグラムをログします。                                                       |
| `training_data`            | (タプル) `model.fit`に渡されるのと同じフォーマット`(X,y)`。 勾配の計算に必要です - `log_gradients`が`True`の場合、これは必須です。       |
| `validation_data`          | (タプル) `model.fit`に渡されるのと同じフォーマット`(X,y)`。 wandbが可視化するためのデータセット。これが設定されている場合、wandbはエポックごとに少数の予測を行い、結果を後で可視化するために保存します。          |

| `generator`                | (generator) wandbが視覚化するためのバリデーションデータを返すジェネレーター。このジェネレーターはタプル`(X,y)`を返すべきです。wandbが特定のデータ例を視覚化するためには、`validate_data`またはgeneratorのどちらかが設定されている必要があります。     |
| `validation_steps`         | (int) `validation_data`がジェネレーターの場合、全バリデーションセットのジェネレーターを実行するステップ数。       |
| `labels`                   | (list) wandbでデータを視覚化している場合、このラベルのリストは、マルチクラス分類器を構築している場合に数値出力を理解しやすい文字列に変換します。バイナリ分類器を作成している場合は、\["label for false", "label for true"]という2つのラベルのリストを渡すことができます。`validate_data`とgeneratorの両方がfalseの場合、これは何もしません。 |
| `predictions`              | (int) 各エポックで視覚化するための予測を行う回数、最大は100。    |

 | `input_type`               | (string) モデル入力のタイプを可視化に役立てます。次のうちのいずれかになります: (`image`, `images`, `segmentation_mask`).  |
| `output_type`              | (string) モデル出力のタイプを可視化に役立てます。次のうちのいずれかになります: (`image`, `images`, `segmentation_mask`).    |
| `log_evaluation`           | (boolean) Trueの場合、各エポックでの検証データとモデルの予測を含む表を保存します。詳細については、`validation_indexes`、`validation_row_processor`、および`output_row_processor`を参照してください。     |
| `class_colors`             | (\[float, float, float]) 入力または出力がセグメンテーションマスクの場合、各クラスに対するrgbタプル（範囲0-1）を含む配列。                  |
| `log_batch_frequency`      | (integer) Noneの場合、コールバックは各エポックでログを記録します。整数に設定されている場合、コールバックは`log_batch_frequency`バッチごとにトレーニングメトリクスを記録します。          |

| `log_best_prefix`          | (string) Noneの場合、追加の要約メトリクスは保存されません。文字列に設定すると、監視されたメトリックとエポックはこの値で前置され、要約メトリクスとして保存されます。   |
| `validation_indexes`       | (\[wandb.data\_types.\_TableLinkMixin]) 検証例に関連付ける順序付けられたインデックスキーのリスト。log\_evaluationがTrueであり、`validation_indexes`が提供されている場合、検証データのテーブルは作成されず、各予測が`TableLinkMixin`で表される行に関連付けられます。このようなキーを取得する最も一般的な方法は、`Table.get_index()`を使用することで、行キーのリストを返します。          |

| `validation_row_processor` | (Callable) 検証データに適用する関数で、データの可視化によく使われます。関数は `ndx`（int）と `row`（dict）を受け取ります。モデルが単一の入力を持つ場合、`row["input"]` は行の入力データになります。それ以外の場合、入力スロットの名前に基づいてキーが割り当てられます。fit関数が単一のターゲットを取る場合、`row["target"]` は行のターゲットデータになります。それ以外の場合、出力スロットの名前に基づいてキーが割り当てられます。例えば、入力データが単一のndarrayである場合でも、データを画像として可視化したい場合は、`lambda ndx, row: {"img": wandb.Image(row["input"])}` を処理関数として提供できます。log\_evaluationがFalseの場合、または `validation_indexes` が存在する場合は無視されます。|
| `output_row_processor`     | (Callable) `validation_row_processor` と同様ですが、モデルの出力に適用されます。`row["output"]` には、モデルの結果が格納されます。          |

| `infer_missing_processors` | (bool) `validation_row_processor` と `output_row_processor` が不足している場合に推定されるかどうかを決定します。デフォルトではTrueです。`labels` が提供されている場合、適切な場所で分類タイプのプロセッサーを推定しようとします。|
| `log_evaluation_frequency` | (int) 評価結果がログに記録される頻度を決定します。デフォルトは 0 (トレーニング終了時のみ)。毎エポックを記録するには 1、エポックごとに 2 を設定します。log\_evaluation が False の場合、影響はありません。|

## よくある質問

### `Keras`のマルチプロセッシングを`wandb`と一緒に使いたい場合、どうすればいいですか？

`use_multiprocessing=True`を設定して、次のようなエラーが表示される場合：

```python
Error('You must call wandb.init() before wandb.config.batch_size')
```

次の手順を試してみてください：

1. `Sequence`クラスの構築で、`wandb.init(group='...')` を追加します。
2. メインプログラムで、`if __name__ == "__main__":` を使用していることを確認し、その後にスクリプトロジックの残りを入れます。