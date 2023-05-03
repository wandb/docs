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
## よくある質問

### `Keras` のマルチプロセッシングと`wandb`をどのように使いますか？

`use_multiprocessing=True` を設定して、以下のようなエラーが表示される場合：

```python
Error('You must call wandb.init() before wandb.config.batch_size')
```

以下の方法を試してください：

1. `Sequence` クラスの構築で、`wandb.init(group='...')` を追加します。
2. メインプログラムで `if __name__ == "__main__":` を使用し、スクリプトのロジックをその中に入れてください。