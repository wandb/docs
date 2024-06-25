---
displayed_sidebar: default
---


# Keras

[**Try in a Colab Notebook here →**](http://wandb.me/intro-keras)

## The Weights & Biases Keras Callbacks

KerasやTensorFlowのユーザー向けに、`wandb` v0.13.4から新しいコールバックを3つ追加しました。従来の`WandbCallback`についてはスクロールダウンしてください。

**`WandbMetricsLogger`** : [Experiment Tracking](https://docs.wandb.ai/guides/track)用のコールバックです。トレーニングと検証用のメトリクス、そしてシステムメトリクスをWeights & Biasesにログします。

**`WandbModelCheckpoint`** : モデルのチェックポイントをWeights & Biasesの[Artifacts](https://docs.wandb.ai/guides/data-and-model-versioning)にログします。

**`WandbEvalCallback`**: このベースコールバックは、モデルの予測をWeights & Biasesの[Tables](https://docs.wandb.ai/guides/tables)にログし、インタラクティブな可視化を可能にします。

これらの新しいコールバックは以下の特徴を持っています：

* Kerasのデザイン哲学を遵守
* 何でも使用するシングルコールバック(`WandbCallback`)の認知負荷を軽減
* Kerasユーザーがニッチなユースケースをサポートするためにコールバックをサブクラス化して簡単に変更可能

## Experiment Tracking with `WandbMetricsLogger`

[**Try in a Colab Notebook here →**](https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb)

`WandbMetricsLogger`は、コールバックメソッド（例：`on_epoch_end`、`on_batch_end`など）が引数として受け取るKerasの`logs`辞書を自動的にログします。

これを使用することで以下が提供されます：

* `model.compile`で定義されたトレインおよび検証メトリクス
* システム（CPU/GPU/TPU）メトリクス
* 学習率（固定値または学習率スケジューラー）

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# 新しいW&B runを初期化
wandb.init(config={"bs": 12})

# WandbMetricsLoggerをmodel.fitに渡す
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbMetricsLogger()]
)
```

**`WandbMetricsLogger` Reference**

| パラメータ | 説明 | 
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | ("epoch", "batch", またはint): "epoch"の場合は各エポックの終了時にメトリクスをログします。"batch"の場合は各バッチ終了時にログします。intの場合、その多くのバッチ後にログします。デフォルトは"epoch"です。 |
| `initial_global_step` | (int): トレーニングをある開始エポックから再開し、学習率スケジューラーが使用される場合、この引数で学習率を正しくログします。これはstep_size*initial_stepとして計算できます。デフォルトは0です。 |

## Model Checkpointing using `WandbModelCheckpoint`

[**Try in a Colab Notebook here →**](https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb)

`WandbModelCheckpoint`コールバックを使用すると、Kerasモデル（`SavedModel`形式）やモデルの重みを定期的に保存し、モデルのバージョン管理用にW&Bに`wandb.Artifact`としてアップロードします。

このコールバックは[`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)をサブクラス化しているため、チェックポイントロジックは親コールバックによって処理されます。

このコールバックの特徴は以下の通りです：

* "モニター"に基づいて「最高のパフォーマンス」を達成したモデルを保存します。
* パフォーマンスに関わらず、各エポックの終了時にモデルを保存します。
* エポックの終了時や一定数のトレーニングバッチ後にモデルを保存します。
* モデルの重みのみを保存するか、全体的なモデルを保存します。
* `SavedModel`形式または`.h5`形式でモデルを保存します。

このコールバックは`WandbMetricsLogger`と一緒に使用するべきです。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

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
    ],
)
```

**`WandbModelCheckpoint` Reference**

| パラメータ | 説明 | 
| ------------------------- |  ---- | 
| `filepath`   | (str): モデルファイルを保存するパス。|  
| `monitor`                 | (str): モニターするメトリクスの名前。         |
| `verbose`                 | (int): 冗長モード、0または1。モード0はサイレントで、モード1はコールバックがアクションを取るたびにメッセージを表示します。   |
| `save_best_only`          | (bool): `save_best_only=True`の場合、モデルが「最高」と見なされたときだけ保存し、モニタリングされる値に従って最新の最高のモデルは上書きされません。     |
| `save_weights_only`       | (bool): Trueの場合、モデルの重みのみを保存。                                            |
| `mode`                    | ("auto", "min", または "max"): val_accには‘max’、val_lossには‘min’など。  |
| `save_weights_only`       | (bool): Trueの場合、モデルの重みのみを保存。                                            |
| `save_freq`               | ("epoch"または int): ‘epoch’を使用する場合、各エポック後にモデルを保存。整数を使用する場合、その多くのバッチ後にモデルを保存。`val_acc`や`val_loss`などの検証メトリクスをモニタリングする場合、`save_freq`は「epoch」に設定する必要があります。検証メトリクスはエポックの終了時にのみ利用可能です。 |
| `options`                 | (str): `save_weights_only`がtrueの場合はオプションの`tf.train.CheckpointOptions`オブジェクト、`save_weights_only`がfalseの場合はオプションの`tf.saved_model.SaveOptions`オブジェクト。    |
| `initial_value_threshold` | (float): モニターされるメトリクスの初期「最良」値。       |

### How to log checkpoints after N epochs?

デフォルト設定（`save_freq="epoch"`）では、コールバックは各エポック後にチェックポイントを作成し、それをアーティファクトとしてアップロードします。`save_freq`に整数を渡すと、その多くのバッチの後にチェックポイントが作成されます。`N`エポック後にチェックポイントを作成するには、トレインデータローダーのカーディナリティを計算し、それを`save_freq`に渡します：

```
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### How to log checkpoints on a TPU Node architecture efficiently?

TPU上でチェックポイントを作成する際に`UnimplementedError: File system scheme '[local]' not implemented`エラーメッセージが発生する場合があります。これは、モデルディレクトリ（`filepath`）がクラウドストレージバケットパス（`gs://bucket-name/...`）を使用する必要があり、このバケットがTPUサーバーからアクセス可能である必要があるためです。しかし、ローカルパスを使用してチェックポイントを作成し、それをArtifactsとしてアップロードすることができます。

```
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## Model Prediction Visualization using `WandbEvalCallback`

[**Try in a Colab Notebook here →**](https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb)

`WandbEvalCallback`は、主にモデルの予測および二次的にデータセットの可視化のためにKerasコールバックを構築するための抽象ベースクラスです。

この抽象コールバックは、データセットやタスクに依存しないため、ベース`WandbEvalCallback`コールバッククラスを継承し、`add_ground_truth`と`add_model_prediction`メソッドを実装して使用します。

`WandbEvalCallback`は便利なメソッドを提供します：

* データと予測の`wandb.Table`インスタンスを作成
* データと予測のテーブルを`wandb.Artifact`としてログ
* `on_train_begin`時にデータテーブルをログ
* `on_epoch_end`時に予測テーブルをログ

例えば、以下の画像分類タスク用の`WandbClfEvalCallback`が実装されています。この例では：

* 検証データ（`data_table`）をW&Bにログ
* 推論を実行し、各エポック終了時に予測（`pred_table`）をW&Bにログ

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback


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

# 新しいW&B runを初期化
wandb.init(config={"hyper": "parameter"})

# コールバックをModel.fitに追加
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

:::info
💡 テーブルはデフォルトでW&Bの[Artifactページ](https://docs.wandb.ai/ref/app/pages/project-page#artifacts-tab)にログされ、[Workspace](https://docs.wandb.ai/ref/app/pages/workspaces)ページではありません。
:::

**`WandbEvalCallback` Reference**

| パラメータ            | 説明                                      |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (リスト) `data_table`の列名リスト |
| `pred_table_columns` | (リスト) `pred_table`の列名リスト |

### メモリフットプリントの削減方法

`on_train_begin`メソッドが呼び出されるときに`data_table`をW&Bにログします。一度W&B Artifactとしてアップロードされると、このテーブルへの参照を取得でき、`data_table_ref`クラス変数を使用してアクセスできます。`data_table_ref`は2次元リストで、`self.data_table_ref[idx][n]`のようにインデックス付けできます。ここで`idx`は行番号、`n`は列番号です。以下の例で使用方法を確認します。

### コールバックをさらにカスタマイズ

`on_train_begin`や`on_epoch_end`メソッドをオーバーライドして、より詳細な制御を行うことができます。`on_train_batch_end`メソッドを実装して、`N`バッチ後にサンプルをログすることもできます。

:::info
💡 `WandbEvalCallback`を継承してモデル予測の可視化コールバックを実装しており、不明な点や修正が必要な点がある場合は、[issueをオープン](https://github.com/wandb/wandb/issues)してください。
:::

## WandbCallback [Legacy]

W&Bライブラリの[`WandbCallback`](https://docs.wandb.ai/ref/python/integrations/keras/wandbcallback)クラスを使用して、`model.fit`で追跡されるすべてのメトリクスと損失値を自動的に保存します。

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Kerasでモデルをセットアップするためのコード

# コールバックをmodel.fitに渡す
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

**使用例**

初めてW&BとKerasを連携させる場合は、この1分間のステップバイステップ動画を参照してください：[Get Started with Keras and Weights & Biases in Less Than a Minute](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M)

より詳細な動画については、[Integrate Weights & Biases with Keras](https://www.youtube.com/watch?v=Bsudo7jbMow&ab_channel=Weights%26Biases)を参照してください。使用されたノートブックの例はこちらにあります：[Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras_pipeline_with_Weights_and_Biases.ipynb)。

:::info
上記のビデオからW&BとKerasのインテグレーション例を[コラボノートブック](http://wandb.me/keras-colab)で試してください。または、[example repo](https://github.com/wandb/examples)でスクリプトを参照してください。その中には、[Fashion MNISTの例](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py)と、それが生成した[W&Bダッシュボード](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs)が含まれています。
:::

`WandbCallback`クラスは、モニターするメトリクスの指定、重みと勾配の追跡、トレーニングおよび検証データに対する予測のログなど、さまざまなログ設定オプションをサポートします。

完全な詳細については[`keras.WandbCallback`](../../ref/python/integrations/keras/wandbcallback.md)のリファレンスドキュメントを参照してください。

`WandbCallback` 

* Kerasによって収集された任意のメトリクスから履歴データを自動的にログ：損失や`keras_model.compile()`に渡されたもの
* `monitor`や`mode`属性により定義された「最佳」のトレーニングステップに関連するサマリーメトリクスを設定します。これによりデフォルトでは最小の`val_loss`を持つエポックが選ばれます。`WandbCallback`はデフォルトで最佳の`epoch`に関連するモデルを保存します。
* 勾配とパラメーターのヒストグラムをオプションでログ
* W&Bが可視化するためにトレーニングデータおよび検証データをオプションで保存可能

**`WandbCallback` Reference**

| 引数                  |                                    |
| -------------------------- | ------------------------------------------- |
| `monitor`                  | (str) モニターするメトリクスの名前。デフォルトは`val_loss`。                                                                   |
| `mode`                     | (str) {`auto`, `min`, `max`}のいずれか。`min` - モニターが最小化されるときにモデルを保存 `max` - モニターが最大化されるときにモデルを保存 `auto` - モデルの保存タイミングを自動判断します（デフォルト）。                                                                                                                                                |
| `