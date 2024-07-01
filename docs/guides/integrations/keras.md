---
displayed_sidebar: default
---

# Keras

[**Try in a Colab Notebook here →**](http://wandb.me/intro-keras)

## The Weights & Biases Keras Callbacks

KerasとTensorFlowユーザー向けに、`wandb` v0.13.4から3つの新しいコールバックを追加しました。従来の`WandbCallback`については、下にスクロールしてください。

**`WandbMetricsLogger`** : [Experiment Tracking](https://docs.wandb.ai/guides/track) 用のこのコールバックを使用してください。トレーニングと検証のメトリクスに加え、システムメトリクスをWeights and Biasesにログします。

**`WandbModelCheckpoint`** : モデルのチェックポイントをWeight and Biases [Artifacts](https://docs.wandb.ai/guides/data-and-model-versioning) にログするために、このコールバックを使用してください。

**`WandbEvalCallback`**: この基本コールバックはモデルの予測をWeights and Biases [Tables](https://docs.wandb.ai/guides/tables) にログし、インタラクティブな可視化を行います。

これらの新しいコールバックは、

* Kerasの設計理念に準拠
* すべての操作を単一のコールバック（`WandbCallback`）で行う場合の認知負荷を軽減
* Kerasユーザーが自分の特定のユースケースに合わせてサブクラス化してコールバックを変更しやすくなります

## `WandbMetricsLogger`を使った実験管理

[**Try in a Colab Notebook here →**](https://github.com/wandb/examples/blob/master/colabs/keras/Use\_WandbMetricLogger\_in\_your\_Keras\_workflow.ipynb)

`WandbMetricsLogger`は、`on_epoch_end`や`on_batch_end`などのコールバックメソッドが引数として受け取るKerasの`logs`辞書を自動的にログします。

これを使用すると、以下の情報が提供されます：

* `model.compile`で定義されたトレーニングと検証のメトリクス
* システム（CPU/GPU/TPU）メトリクス
* 固定値または学習率スケジューラの学習率

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# 新しいW&B runを初期化する
wandb.init(config={"bs": 12})

# WandbMetricsLoggerをmodel.fitに渡す
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbMetricsLogger()]
)
```

**`WandbMetricsLogger` 参照**

| パラメータ | 説明 | 
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | ("epoch", "batch", または int): "epoch"の場合、各エポックの終了時にメトリクスをログします。"batch"の場合、各バッチの終了時にメトリクスをログします。intの場合、そのバッチ数の終了時にメトリクスをログします。デフォルトは"epoch"です。                                 |
| `initial_global_step` | (int): 学習率がスケジューラを使用している場合に、初期のエポックからトレーニングを再開する際に学習率を正しくログするために使用します。これはstep_size * initial_stepとして計算できます。デフォルトは0です。 |

## `WandbModelCheckpoint`を使用したモデルのチェックポイント作成

[**Try in a Colab Notebook here →**](https://github.com/wandb/examples/blob/master/colabs/keras/Use\_WandbModelCheckpoint\_in\_your\_Keras\_workflow.ipynb)

`WandbModelCheckpoint`コールバックを使用して、Kerasモデル（`SavedModel`形式）やモデルの重みを定期的に保存し、モデルのバージョン管理のためにW&Bの`wandb.Artifact`としてアップロードします。

このコールバックは、[`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/ModelCheckpoint) からサブクラス化されているため、チェックポイントのロジックは親コールバックが処理します。

このコールバックは次の機能を提供します：

* "モニター"に基づいて"最高のパフォーマンス"を達成したモデルを保存します。
* パフォーマンスに関わらず、各エポックの終了時にモデルを保存します。
* エポックの終了時または特定のトレーニングバッチ数後にモデルを保存します。
* モデルの重みのみを保存するか、モデル全体を保存します。
* モデルをSavedModel形式または`.h5`形式で保存します。

このコールバックは`WandbMetricsLogger`と一緒に使用する必要があります。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# 新しいW&Bのrunを初期化する
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

**`WandbModelCheckpoint` 参照**

| Parameter | Description | 
| ------------------------- |  ---- | 
| `filepath`   | (str): モデルファイルを保存するパス。|  
| `monitor`                 | (str): 監視するメトリック名。         |
| `verbose`                 | (int): 冗長モード、0または1。モード0はサイレントで、モード1はコールバックがアクションを実行する際にメッセージを表示します。   |
| `save_best_only`          | (bool): `save_best_only=True`の場合、モデルが"最高"と見なされた場合にのみ保存され、監視される量に基づいて最新の最高のモデルは上書きされません。     |
| `save_weights_only`       | (bool): Trueの場合、モデルの重みだけが保存されます。                                            |
| `mode`                    | ("auto", "min", または "max"): val\_accの場合、これは‘max’であり、val\_lossの場合、これは‘min’である必要があります。  |
| `save_weights_only`       | (bool): Trueの場合、モデルの重みだけが保存されます。                                            |
| `save_freq`               | ("epoch" または int): ‘epoch’を使用すると、各エポックの後にモデルが保存されます。整数を使用すると、そのバッチ数の終了時にモデルが保存されます。`val_acc`や`val_loss`などの検証メトリクスを監視する場合、`save_freq`はエポックの終了時にのみ利用可能なため"epoch"に設定する必要があります。 |
| `options`                 | (str): `save_weights_only`がtrueの場合はオプションの`tf.train.CheckpointOptions`オブジェクト、`save_weights_only`がfalseの場合はオプションの`tf.saved_model.SaveOptions`オブジェクト。    |
| `initial_value_threshold` | (float): 監視するメトリックの初期"最高"値。       |

### Nエポック後にチェックポイントをログする方法は？

デフォルトでは（`save_freq="epoch"`）、コールバックは各エポックの後にチェックポイントを作成し、それをアーティファクトとしてアップロードします。`save_freq`に整数を渡すと、そのバッチ数後にチェックポイントが作成されます。 `N`エポック後にチェックポイントを作成するには、トレインデータローダーのカーディナリティを計算し、それを`save_freq`に渡します：

```
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### TPUノードアーキテクチャーで効率的にチェックポイントをログする方法は？

TPUでチェックポイントを作成する際に`UnimplementedError: File system scheme '[local]' not implemented`エラーが発生することがあります。これは、モデルディレクトリ（`filepath`）がクラウドストレージバケットのパス（`gs://bucket-name/...`）を使用する必要があり、このバケットがTPUサーバーからアクセス可能である必要があるためです。 ただし、ローカルパスを使用してチェックポイントを作成し、それをArtifactsとしてアップロードすることは可能です。

```
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## `WandbEvalCallback`を使用したモデル予測の可視化

[**Try in a Colab Notebook here →**](https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb)

`WandbEvalCallback`は、主にモデル予測およびデータセットの可視化のためにKerasコールバックを構築するための抽象基本クラスです。

この抽象コールバックは、データセットやタスクに依存しません。これを使用するには、この基本`WandbEvalCallback`コールバッククラスから継承し、`add_ground_truth`および`add_model_prediction`メソッドを実装します。

`WandbEvalCallback`は以下の便利なメソッドを提供するユーティリティクラスです:

* データと予測の`wandb.Table`インスタンスを作成
* データと予測テーブルを`wandb.Artifact`としてログ
* `on_train_begin`でデータテーブルをログ
* `on_epoch_end`で予測テーブルをログ

例えば、以下の画像分類タスクのために`WandbClfEvalCallback`を実装しました。この例のコールバックは：

* W&Bに検証データ（`data_table`）をログ
* 推論を行い、各エポックの終了時に予測（`pred_table`）をW&Bにログ

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback


# モデル予測可視化コールバックを実装する
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

:::info
💡 テーブルはデフォルトでW&Bの[Artifact page](https://docs.wandb.ai/ref/app/pages/project-page#artifacts-tab)にログされ、[Workspace](https://docs.wandb.ai/ref/app/pages/workspaces)ページには表示されません。
:::

**`WandbEvalCallback` 参照**

| Parameter | Description                                      |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (list) `data_table`のカラム名のリスト |
| `pred_table_columns` | (list) `pred_table`のカラム名のリスト |

### メモリ使用量はどのように削減されるか？

`on_train_begin`メソッドが呼び出されたときに`data_table`をW&Bにログします。一度W&B Artifactとしてアップロードされると、このテーブルへの参照を得ることができ、`data_table_ref`クラス変数を使用してアクセスできます。`data_table_ref`は2Dリストで、`self.data_table_ref[idx][n]`のようにインデックスすることができます。ここで`idx`は行番号を表し、`n`は列番号です。以下の例で使用方法を見てみましょう。

### コールバックをさらにカスタマイズする

`on_train_begin`や`on_epoch_end`メソッドをオーバーライドして、より詳細に制御することができます。`N`バッチ後にサンプルをログしたい場合は、`on_train_batch_end`メソッドを実装します。

:::info
💡 `WandbEvalCallback`から継承して、モデル予測の可視化コールバックを実装している場合、何か明確にしたいことがあるか修正が必要な場合は、[issue](https://github.com/wandb/wandb/issues)を開いてお知らせください。
:::

## WandbCallback [Legacy]

W&Bライブラリの[`WandbCallback`](https://docs.wandb.ai/ref/python/integrations/keras/wandbcallback)クラスを使用して、`model.fit`で追跡されたすべてのメトリクスと損失値を自動的に保存します。

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Kerasでモデルをセットアップするコード

# コールバックをmodel.fitに渡す
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

**使用例**

W&BとKerasの統合が初めての場合は、この1分間のステップバイステップビデオを参照してください：[Get Started with Keras and Weights & Biases in Less Than a Minute](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M)

より詳細なビデオについては、[Integrate Weights & Biases with Keras](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab\_channel=Weights%26Biases)をご覧ください。使用されたノートブックの例はこちらにあります：[Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras\_pipeline\_with\_Weights\_and\_Biases.ipynb)。

:::info
上記のビデオ例を[Colabノートブック](http://wandb.me/keras-colab)で試してみてください。 または[example repo](https://github.com/wandb/examples)を参照して、スクリプトや[Fashion MNIST example](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py)および生成された[W&Bダッシュボード](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs)を確認してください。
:::

`WandbCallback`クラスは、監視するメトリクスの指定、重みおよび勾配の追跡、トレーニングデータおよび検証データの予測のログなど、さまざまなログ設定オプションをサポートしています。

完全な詳細については、[`keras.WandbCallback` のリファレンスドキュメント](../../ref/python/integrations/keras/wandbcallback.md)を参照してください。

`WandbCallback`は次のことを行います：

* kerasによって収集された任意のメトリクスからの履歴データを自動的にログします：損失および`keras_model.compile()`に渡されたもの
*  "best"なトレーニングステップに関連するrunのサマリーメトリクスを設定します。"best"は`monitor`および`mode`属性によって定義されます。これはデフォルトで最小の`val_loss`を持つエポックです。 `WandbCallback`はデフォルトで最高のエポックに関連するモデルを保存します
* 勾配およびパラメーターのヒストグラムをオプションでログできます
* wandbで可視化するためのトレーニングおよび検証データをオプションで保存できます

**`WandbCallback` 参照**

| Arguments                  |                                    |
| -------------------------- | ------------------------------------------- |
| `monitor`                  | (str) 監視するメトリクス名。デフォルトは `val_loss`。                                                                   |
| `mode`                     | (str) `{`auto`, `min`, `max`}` のいずれか。一番小さい monitor 値の時にモデルを保存する - `min`、一番大きい monitor 値の時にモデルを保存する - `max`、 自動的にモデルを保存する - `auto`（デフォルト）。                                                                                                                                                |
| `save_model`               | True - monitor が過去のエポックをすべて超えるとモデルを保存する。False - モデルを保存しない。                                       |
| `save_graph`               | (boolean) True の場合、モデルグラフを wandb に保存します（デフォルトは True）。                                                           |
| `save_weights_only`        | (boolean) True の場合、モデルの重みだけが保存されます（`model.save_weights(filepath)`）、そうでない場合はフルモデルが保存されます（`model.save(filepath)`）。   |
| `log_weights`              | (boolean) True の場合、モデルの層の重みのヒストグラムが保存されます。                                                |
| `log_gradients`            | (boolean) True の場合、トレーニング勾配のヒストグラムがログされます。                                                       |
| `training_data`            | (tuple) `(X,y)` のフォーマット。勾配の計算用に必要です - これは`log_gradients`がTrue の場合必須です。       |
| `validation_data`          | (tuple) `(X,y)` のフォーマット。wandb が視覚化するためのデータセット。これが設定されている場合、毎エポック後に小さな数の予測を行い、将来の視覚化のために結果を保存します。          |
| `generator`                | (generator) wandb が視覚化するための検証データを返すジェネレータ。このジェネレータは `(X,y)` のタプルを返すべきです。 `validate_data` または generator のどちらかが設定されるべきです。     |
| `validation_steps`         | (int) `validation_data` がジェネレータの場合、ジェネレータをフル検証セットのために何ステップ走らせるか。       |
| `labels`                   | (list) データを wandb で視覚化する場合、このラベルのリストは数値出力を理解しやすい文字列に変換します。バイナリ分類器を構築する場合、["false のラベル", "true のラベル"] のリストを渡すことができます。`validate_data` や generator のいずれも false の場合、効果はありません。    |
| `predictions`              | (int) 各エポックの視覚化のために行う予測の数、最大は 100。    |
| `input_type`               | (string) 視覚化を助けるためのモデル入力のタイプ。以下のいずれか: (`image`, `images`, `segmentation_mask`)。  |
| `output_type`              | (string) 視覚化を助けるためのモデル出力のタイプ。以下のいずれか: (`image`, `images`, `segmentation_mask`)。    |
| `log_evaluation`           | (boolean) True の場合、各エポックでの検証データとモデルの予測を含むテーブルが保存されます。 詳細については `validation_indexes`、 `validation_row_processor` 、`output_row_processor` を参照してください。     |
| `class_colors`             | (\[float, float, float]) 入力または出力がセグメンテーションマスクである場合、各クラスの RGB タプル (0-1 の範囲) を含む配列。                  |
| `log_batch_frequency`      | (integer) None の場合、コールバックは各エポックごとにログします。整数に設定すると、`log_batch_frequency` バッチごとにトレーニングメトリクスがログされます。          |
| `log_best_prefix`          | (string) Noneの場合、追加のサマリーメトリクスは保存されません。文字列に設定すると、監視されたメトリクスとエポックがこの値で先頭に付けられ、サマリーメトリクスとして保存されます。   |
| `validation_indexes`       | (\[wandb.data\_types.\_TableLinkMixin]) 各検証例と関連するインデックスキーの順序付きリスト。 log\_evaluation が True であり `validation_indexes` が提供されている場合、検証データのテーブルは作成されず、各予測は `TableLinkMixin` によって表される行と関連付けられます。これらのキーを取得する同様の方法は `Table.get_index()` で返される行キーのリストを使用します。          |
| `validation_row_processor` | (Callable) 検証データに適用される関数であり、視覚化によく使用されます。この関数は `ndx` (int) と `row` (dict) を受け取ります。モデルが単一の入力を持つ場合は `row["input"]` がその行の入力データになります。それ以外の場合は、入力スロットの名前に基づいてキー付けされます。フィット関数が単一のターゲットを取る場合、`row["target"]` はその行のターゲットデータになります。それ以外の場合は、出力スロットの名前に基づいてキー付けされます。例えば、入力データが単一の ndarry であるが、データを画像として視覚化したい場合、`lambda ndx, row: {"img": wandb.Image(row["input"])}` をプロセッサとして提供できます。 log\_evaluation が False または `validation_indexes` が存在する場合は無視されます。 |
| `output_row_processor`     | (Callable) `validation_row_processor` と同じですが、モデルの出力に適用されます。`row["output"]` はモデル出力の結果を含みます。          |
| `infer_missing_processors` | (bool) `validation_row_processor` および `output_row_processor` が欠けている場合に推測するかどうかを決定します。 デフォルトは True です。 ラベルが提供されている場合、適切な場所に分類タイププロセッサを推測しようとします。      |
| `log_evaluation_frequency` | (int) 評価結果がログされる頻度を決定します。 デフォルトは 0 (トレーニング終了時のみログ)。 毎エポックログする場合は 1 を設定し、他のエポック毎の場合は 2 を設定し、等々。 log\_evaluation が False の場合は効果がありません。    |

## よくある質問

### `Keras` のマルチプロセッシングを `wandb` で使用するにはどうすればよいですか？

`use_multiprocessing=True` を設定していると、次のようなエラーが発生する場合があります：

```python
Error("You must call wandb.init() before wandb.config.batch_size")
```

その場合は次のようにしてみてください：

