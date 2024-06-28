---
displayed_sidebar: default
---


# Keras

[**Colabノートブックで試す →**](http://wandb.me/intro-keras)

## Weights & Biases の Keras コールバック

`wandb` v0.13.4から、KerasおよびTensorFlowユーザー向けに3つの新しいコールバックを追加しました。レガシー`WandbCallback`については下にスクロールしてください。

**`WandbMetricsLogger`** : このコールバックを使って、[実験管理](https://docs.wandb.ai/guides/track)を行います。トレーニングとバリデーションのメトリクス、システムメトリクスをWeights & Biases にログします。

**`WandbModelCheckpoint`** : このコールバックを使って、モデルのチェックポイントをWeights & Biases の [Artifacts](https://docs.wandb.ai/guides/data-and-model-versioning) にログします。

**`WandbEvalCallback`**: このベースコールバックは、モデルの予測をWeights & Biases の [Tables](https://docs.wandb.ai/guides/tables) にログし、インタラクティブな可視化を可能にします。

これらの新しいコールバックは、

* Kerasのデザイン哲学に従っています
* すべてを単一のコールバック (`WandbCallback`) で処理する際の認知負荷を軽減します
* Kerasユーザーが自分のニッチなユースケースをサポートするためにコールバックをサブクラス化して簡単に変更できるようにします。

## `WandbMetricsLogger` を使った実験管理

[**Colabノートブックで試す →**](https://github.com/wandb/examples/blob/master/colabs/keras/Use\_WandbMetricLogger\_in\_your\_Keras\_workflow.ipynb)

`WandbMetricsLogger`は、`on_epoch_end`、`on_batch_end`などのコールバックメソッドが引数として受け取るKerasの`logs`辞書を自動的にログします。

これを使うことで以下が提供されます：

* `model.compile`で定義されたトレーニングとバリデーションのメトリクス
* システム（CPU/GPU/TPU）メトリクス
* 学習率（固定値または学習率スケジューラのどちらでも）

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

**`WandbMetricsLogger` リファレンス**

| パラメータ | 説明 | 
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | ("epoch", "batch", または int): "epoch"の場合、各エポックの終わりにメトリクスをログします。"batch"の場合、各バッチの終わりにメトリクスをログします。整数の場合はその多くのバッチ後にメトリクスをログします。デフォルトは "epoch"。                                 |
| `initial_global_step` | (int): トレーニングを初期エポックから再開する際に学習率を正しくログするために使用する引数です。これは step_size * initial_step として計算できます。デフォルトは0。 |

## `WandbModelCheckpoint` を使ったモデルのチェックポイント

[**Colabノートブックで試す →**](https://github.com/wandb/examples/blob/master/colabs/keras/Use\_WandbModelCheckpoint\_in\_your\_Keras\_workflow.ipynb)

`WandbModelCheckpoint` コールバックを使用して、Keras モデル（`SavedModel`形式）またはモデルの重みを定期的に保存し、モデルのバージョン管理のために W&B に `wandb.Artifact` としてアップロードします。

このコールバックは、[`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/ModelCheckpoint) から継承されているため、チェックポイントのロジックは親コールバックによって処理されます。

このコールバックは以下の機能を提供します：

* "モニター"に基づいて "最高のパフォーマンス"を達成したモデルを保存します。
* パフォーマンスに関係なく各エポックの終わりにモデルを保存します。
* エポックの終わりまたは一定数のトレーニングバッチ後にモデルを保存します。
* モデルの重みのみを保存するか、モデル全体を保存します。
* モデルをSavedModel形式または `.h5`形式で保存します。

このコールバックは `WandbMetricsLogger` と併用する必要があります。

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

**`WandbModelCheckpoint` リファレンス**

| パラメータ | 説明 | 
| ------------------------- |  ---- | 
| `filepath`   | (str): 保存するモデルファイルのパス。|  
| `monitor`                 | (str): モニターするメトリクスの名前。         |
| `verbose`                 | (int): 冗長モード、0または1。モード0はサイレント、モード1はコールバックがアクションを取るときにメッセージを表示。   |
| `save_best_only`          | (bool): `save_best_only=True`の場合、モデルが "ベスト" と見なされたときのみ保存し、監視されている量（`monitor`）に基づいて最新のベストモデルは上書きされません。     |
| `save_weights_only`       | (bool): Trueの場合、モデルの重みのみを保存します。                                            |
| `mode`                    | ("auto", "min", または "max"): val\_acc の場合は 'max'、val\_loss の場合は 'min' など。 |
| `save_freq`               | ("epoch" または int): ‘epoch’ を使用する場合、コールバックは各エポックの後にモデルを保存します。整数を使用する場合、コールバックはその多くのバッチの後にモデルを保存します。`val_acc` や `val_loss` などのバリデーションメトリクスを監視する場合は、`save_freq` を "epoch" に設定する必要があります。これらのメトリクスはエポックの終わりにのみ利用可能です。 |
| `options`                 | (str): `save_weights_only` が true の場合はオプションの `tf.train.CheckpointOptions` オブジェクト、`save_weights_only` が false の場合はオプションの `tf.saved_model.SaveOptions` オブジェクト。    |
| `initial_value_threshold` | (float): 監視されるメトリクスの初期 "ベスト" 値です。 |

### Nエポック後にチェックポイントをログする方法

デフォルトでは (`save_freq="epoch"`) このコールバックは各エポックの後にチェックポイントを作成し、それをアーティファクトとしてアップロードします。`save_freq` に整数を渡すと、その多くのバッチ後にチェックポイントが作成されます。`N`エポック後にチェックポイントを作成するには、トレインデータローダーのカーディナリティを計算して `save_freq` に渡します:

```
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### TPU ノードアーキテクチャーで効率的にチェックポイントをログする方法

TPU 上でチェックポイントを作成する際に、`UnimplementedError: File system scheme '[local]' not implemented` というエラーメッセージが表示される場合があります。これは、モデルディレクトリ（`filepath`）がクラウドストレージバケットのパス（`gs://bucket-name/...`）を使用する必要があり、このバケットは TPU サーバーからアクセス可能である必要があるためです。しかし、ローカルパスを使用してチェックポイントを作成し、それをアーティファクトとしてアップロードすることもできます。

```
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## `WandbEvalCallback` を使ったモデル予測の可視化

[**Colabノートブックで試す →**](https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb)

`WandbEvalCallback` は、主にモデル予測およびデータセットの可視化のための Keras コールバックを作成するための抽象基底クラスです。

この抽象コールバックはデータセットやタスクに依存しません。これを使用するには、このベース `WandbEvalCallback` コールバッククラスを継承し、`add_ground_truth` メソッドと `add_model_prediction` メソッドを実装します。

`WandbEvalCallback` は便利なメソッドを提供します：

* データと予測の `wandb.Table` インスタンスを作成します
* データと予測のTablesを `wandb.Artifact` としてログします
* トレーニング開始時にデータテーブルをログします `on_train_begin`
* エポック終了時に予測テーブルをログします `on_epoch_end`

例えば、以下の画像分類タスク向けに `WandbClfEvalCallback` を実装しました。この例のコールバックは：

* バリデーションデータ (`data_table`) をW&Bにログします
* 推論を行い、各エポック終了時に予測 (`pred_table`)をW&Bにログします

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback

# モデル予測の可視化コールバックを実装する
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

:::情報
💡 テーブルはデフォルトでW&Bの [Artifactページ](https://docs.wandb.ai/ref/app/pages/project-page#artifacts-tab) にログされ、[Workspace](https://docs.wandb.ai/ref/app/pages/workspaces)ページではありません。
:::

**`WandbEvalCallback` リファレンス**

| パラメータ            | 説明                                      |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (リスト) `data_table` の列名リスト |
| `pred_table_columns` | (リスト) `pred_table` の列名リスト |

### メモリフットプリントの削減方法

`on_train_begin`メソッドが呼び出されるときに、`data_table`をW&Bにログします。一度それがW&B Artifactとしてアップロードされると、このテーブルの参照が取得でき、`data_table_ref` クラス変数を使用してアクセスできます。`data_table_ref` は2次元リストで、`self.data_table_ref[idx][n]`としてインデックスをつけることができます。ここで `idx` は行番号、`n` は列番号です。以下の例で使い方を見てみましょう。

### コールバックをさらにカスタマイズする

`on_train_begin` または `on_epoch_end` メソッドをオーバーライドして、より細かい制御を行うこともできます。`N` バッチ後にサンプルをログしたい場合は、`on_train_batch_end` メソッドを実装できます。

:::情報
💡 `WandbEvalCallback`を継承してモデル予測の可視化用のコールバックを実装している場合、何か不明な点や修正が必要な点がありましたら、[issueを開く](https://github.com/wandb/wandb/issues)ことでお知らせください。
:::


## WandbCallback [Legacy]

W&Bライブラリの[`WandbCallback`](https://docs.wandb.ai/ref/python/integrations/keras/wandbcallback)クラスを使って、`model.fit`で追跡されるすべてのメトリクスと損失値を自動で保存します。

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Kerasでモデルを設定するコード

# コールバックをmodel.fitに渡します
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

**使用例**

初めてW&BとKerasをインテグレーションする場合は、こちらの1分間のステップバイステップ動画をご覧ください: [Get Started with Keras and Weights & Biases in Less Than a Minute](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M)

もっと詳しい動画については、こちらをご覧ください: [Integrate Weights & Biases with Keras](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab\_channel=Weights%26Biases)。使用されたノートブックの例はこちらで見つけることができます: [Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras\_pipeline\_with\_Weights\_and\_Biases.ipynb).

:::info
上記のビデオのW&BとKerasのインテグレーションの例を[Colabノートブック](http://wandb.me/keras-colab)で試してみてください。もしくは、[example repo](https://github.com/wandb/examples)を参照してスクリプトを確認し、[Fashion MNISTの例](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py)とそれが生成する[W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs)を見てください。
:::

この`WandbCallback`クラスは、メトリクスを監視するための設定、重みと勾配の追跡、トレーニングデータと検証データの予測のログを含む、さまざまなログ設定オプションをサポートしています。

詳細については、[the reference documentation for the `keras.WandbCallback`](../../ref/python/integrations/keras/wandbcallback.md)を参照してください。

`WandbCallback`は

* Kerasで収集された任意のメトリクスから履歴データを自動的にログします: 損失と`keras_model.compile()`に渡された他のもの
* "best" トレーニングステップに関連するrunのサマリーメトリクスを設定します。"best"は`monitor`と`mode`属性によって定義され、デフォルトでは最小の`val_loss`を持つエポックです。`WandbCallback`はデフォルトで最も優れたエポックに関連するモデルを保存します
* 勾配とパラメータのヒストグラムをオプションでログします
* wandbが可視化するトレーニングデータと検証データをオプションで保存することもできます

**`WandbCallback` リファレンス**

| 引数                       |                                     |
| -------------------------- | ------------------------------------------------ |
| `monitor`                  | (str) 監視するメトリクスの名前。デフォルトは`val_loss`。     |
| `mode`                     | (str) {`auto`, `min`, `max`}のいずれか。`min` - monitorが最小化されたときにモデルを保存する `max` - monitorが最大化されたときにモデルを保存する `auto` - モデルを保存するタイミングを自動で推測する（デフォルト）。        |
| `save_model`               | True - monitorが全ての以前のエポックを上回ったときにモデルを保存する False - モデルを保存しない         |
| `save_graph`               | (boolean) Trueの場合モデルのグラフをwandbに保存します（デフォルトはTrue）。          |
| `save_weights_only`        | (boolean) Trueならモデルの重みだけを保存する（`model.save_weights(filepath)`）それ以外ならフルモデルを保存する（`model.save(filepath)`）。    |
| `log_weights`              | (boolean) Trueならモデルの各層の重みのヒストグラムを保存する。          |
| `log_gradients`            | (boolean) Trueならトレーニング勾配のヒストグラムをログします。          |
| `training_data`            | (tuple) `model.fit`に渡されるのと同じ形式の`(X,y)`。これは勾配の計算に必要です - `log_gradients`がTrueの場合は必須となります。          |
| `validation_data`          | (tuple) `model.fit`に渡すのと同じ形式の`(X,y)`。wandbによる可視化のためのデータセットです。これが設定されている場合、各エポックでwandbは少数の予測を行い、その結果を後で可視化するために保存します。          |
| `generator`                | (generator) wandbが可視化する検証データを返すジェネレーター。このジェネレーターは`(X,y)`のタプルを返す必要があります。特定のデータ例を可視化するために、`validate_data`かgeneratorのどちらかを設定する必要があります。          |
| `validation_steps`         | (int) `validation_data`がジェネレーターである場合、完全な検証セットのためにジェネレーターを何ステップ実行するか。          |
| `labels`                   | (list) wandbでデータを可視化する場合、このラベルのリストを使用して、数値出力を理解可能な文字列に変換します。多クラス分類器を構築している場合は、2つのラベルのリスト\["falseのラベル", "trueのラベル"]を渡すことができます。`validate_data`とgeneratorの両方がfalseの場合、これは何もしません。          |
| `predictions`              | (int) 各エポックで可視化のために行う予測の数、最大100。          |
| `input_type`               | (string) 可視化を助けるためのモデル入力のタイプ。以下のいずれか: (`image`, `images`, `segmentation_mask`)。          |
| `output_type`              | (string) 可視化を助けるためのモデル出力のタイプ。以下のいずれか: (`image`, `images`, `segmentation_mask`)。          |
| `log_evaluation`           | (boolean) Trueなら、各エポックでの検証データとモデルの予測を含むテーブルを保存します。詳細については`validation_indexes`、`validation_row_processor`、および`output_row_processor`を参照してください。          |
| `class_colors`             | (\[float, float, float]) 入力または出力がセグメンテーションマスクの場合、各クラスに対するrgbタプル（範囲は0-1）の配列。          |
| `log_batch_frequency`      | (integer) Noneの場合、コールバックは各エポックをログします。整数を設定すると、`log_batch_frequency`バッチごとにトレーニングメトリクスをログします。          |
| `log_best_prefix`          | (string) Noneなら追加のサマリーメトリクスは保存されません。文字列を設定した場合、監視するメトリクスとエポックはこの値でプレフィックスされ、サマリーメトリクスとして保存されます。          |
| `validation_indexes`       | (\[wandb.data\_types.\_TableLinkMixin]) 各検証例に関連付けるインデックスキーの順序付きリスト。log\_evaluationがTrueであり、`validation_indexes`が提供されている場合、検証データのテーブルは作成されず、代わりに各予測が`TableLinkMixin`によって表される行に関連付けられます。こうしたキーを取得する最も一般的な方法は、`Table.get_index()`を使用することで、このメソッドは行キーのリストを返します。          |
| `validation_row_processor` | (Callable) 検証データに適用する関数、通常はデータを可視化するために使用されます。この関数は`ndx`（int）と`row`（dict）を受け取ります。モデルが単一の入力を持つ場合、`row["input"]`はその行の入力データになります。それ以外の場合、入力スロットの名前に基づいてキー付けされます。fit関数が単一のターゲットを取る場合、`row["target"]`はその行のターゲットデータになります。それ以外の場合、出力スロットの名前に基づいてキー付けされます。例えば、入力データが単一のndarrayであるが、データを画像として可視化する場合、`lambda ndx, row: {"img": wandb.Image(row["input"])}`をプロセッサとして提供できます。log\_evaluationがFalseまたは`validation_indexes`が存在する場合は無視されます。          |
| `output_row_processor`     | (Callable) `validation_row_processor`と同様ですが、モデルの出力に適用されます。`row["output"]`はモデル出力の結果を含みます。          |
| `infer_missing_processors` | (bool) `validation_row_processor`と`output_row_processor`が欠落している場合に推測するかどうかを決定します。デフォルトはTrue。ラベルが提供されている場合、適切な場所で分類型プロセッサを推測しようとします。          |
| `log_evaluation_frequency` | (int) 評価結果がログされる頻度を決定します。デフォルトは0（トレーニングの最後にのみ）。1に設定すると各エポックごとにログされ、2に設定すると隔エポックごとにログされます。log\_evaluationがFalseの場合は効果がありません。         |

## よくある質問

### `Keras`の並行処理を`wandb`と一緒に使うにはどうすればいいですか？

`use_multiprocessing=True`を設定しているときに、次のようなエラーが表示される場合:

```python
Error("You must call wandb.init() before wandb.config.batch_size")
```

このようにしてください:

1. `Sequence`クラスの構築時に、`wandb.init(group='...')`を追加します
2. メインプログラムで、`if __name__ == "__main__":`を使い、スクリプトのロジックをその中に入れてください。