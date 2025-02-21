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

W&B には、`wandb` v0.13.4 から利用できる Keras 用の 3 つのコールバックがあります。従来の `WandbCallback` については、下にスクロールしてください。

- **`WandbMetricsLogger`**: [実験管理]({{< relref path="/guides/models/track" lang="ja" >}}) にはこのコールバックを使用します。トレーニングと検証の メトリクス を、システム メトリクス とともに Weights & Biases に記録します。

- **`WandbModelCheckpoint`**: このコールバックを使用して、モデル の チェックポイント を Weight and Biases [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) に記録します。

- **`WandbEvalCallback`**: このベース コールバックは、モデル の 予測 を Weights and Biases [Tables]({{< relref path="/guides/core/tables/" lang="ja" >}}) に記録し、インタラクティブな 可視化 を行います。

これらの新しいコールバック：

* Keras の設計思想に準拠しています。
* すべてを 1 つのコールバック (`WandbCallback`) で行うことによる認知負荷を軽減します。
* Keras ユーザー が、ニッチな ユースケース をサポートするために、サブクラス化してコールバックを簡単に変更できるようにします。

## `WandbMetricsLogger` で 実験 を追跡する

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}

`WandbMetricsLogger` は、`on_epoch_end`、`on_batch_end` などのコールバック メソッド が 引数 として受け取る Keras の `logs` 辞書 を自動的に記録します。

これにより、以下が追跡されます。

* `model.compile` で定義されたトレーニングと検証の メトリクス 。
* システム (CPU/GPU/TPU) メトリクス。
* 学習率 (固定値と学習率スケジューラーの両方)。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# 新しい W&B の run を初期化します
wandb.init(config={"bs": 12})

# WandbMetricsLogger を model.fit に渡します
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbMetricsLogger()]
)
```

### `WandbMetricsLogger` リファレンス

| パラメータ | 説明 | 
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq` | (`epoch`、`batch`、または `int`): `epoch` の場合、各 エポック の最後に メトリクス を記録します。`batch` の場合、各 バッチ の最後に メトリクス を記録します。`int` の場合、指定された数の バッチ の最後に メトリクス を記録します。デフォルトは `epoch` です。 |
| `initial_global_step` | (int): 学習率スケジューラーを使用している場合に、トレーニングを初期 エポック から再開するときに、学習率を正しく記録するには、この 引数 を使用します。これは、step_size * initial_step として計算できます。デフォルトは 0 です。 |

## `WandbModelCheckpoint` を使用してモデルを チェックポイント する

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}

`WandbModelCheckpoint` コールバックを使用して、Keras モデル (`SavedModel` 形式) またはモデルの重みを定期的に保存し、モデル の バージョン管理 用に `wandb.Artifact` として W&B にアップロードします。

このコールバックは [`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) からサブクラス化されているため、チェックポイント ロジックは親コールバックによって処理されます。

このコールバックは以下を保存します。

* モニター に基づいて最高のパフォーマンスを達成したモデル。
* パフォーマンスに関係なく、すべての エポック の最後にモデル。
* エポック の最後、または固定数のトレーニング バッチ の後にモデル。
* モデルの重みのみ、またはモデル全体。
* `SavedModel` 形式または `.h5` 形式のいずれかのモデル。

このコールバックは `WandbMetricsLogger` と組み合わせて使用します。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# 新しい W&B の run を初期化します
wandb.init(config={"bs": 12})

# WandbModelCheckpoint を model.fit に渡します
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
| `filepath` | (str): モード ファイル を保存するパス。|
| `monitor` | (str): モニター する メトリクス 名。|
| `verbose` | (int): 冗長モード、0 または 1。モード 0 はサイレントで、モード 1 はコールバックがアクションを実行するときにメッセージを表示します。 |
| `save_best_only` | (Boolean): `save_best_only=True` の場合、`monitor` および `mode` 属性で定義されたものに従って、最新のモデルまたは最適と見なされるモデルのみを保存します。 |
| `save_weights_only` | (Boolean): True の場合、モデル の重みのみを保存します。|
| `mode` | (`auto`、`min`、または `max`): `val_acc` の場合は `max` に、`val_loss` の場合は `min` に設定します | |
| `save_freq` | ("epoch" または int): 「epoch」を使用すると、コールバックは各 エポック の後にモデルを保存します。整数を使用すると、コールバックはこの数の バッチ の終わりにモデルを保存します。`val_acc` や `val_loss` などの検証 メトリクス を モニター する場合、これらの メトリクス は エポック の最後にのみ使用できるため、`save_freq` を「epoch」に設定する必要があることに注意してください。 |
| `options` | (str): `save_weights_only` が true の場合は、オプションの `tf.train.CheckpointOptions` オブジェクト、`save_weights_only` が false の場合は、オプションの `tf.saved_model.SaveOptions` オブジェクト。|
| `initial_value_threshold` | (float): モニター する メトリクス の浮動小数点型の初期「最適」値。|

### N エポック 後に チェックポイント を記録する

デフォルト (`save_freq="epoch"`) では、コールバックは各 エポック の後に チェックポイント を作成し、それを Artifact としてアップロードします。特定の数の バッチ の後に チェックポイント を作成するには、`save_freq` を整数に設定します。`N` エポック 後に チェックポイント を作成するには、`train` データローダー のカーディナリティを計算し、それを `save_freq` に渡します。

```python
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### TPU アーキテクチャー で チェックポイント を効率的に記録する

TPU で チェックポイント を作成しているときに、`UnimplementedError: File system scheme '[local]' not implemented` というエラー メッセージ が表示される場合があります。これは、モデル ディレクトリー (`filepath`) がクラウド ストレージ バケット パス (`gs://bucket-name/...`) を使用する必要があり、このバケットが TPU サーバー からアクセスできる必要があるために発生します。ただし、チェックポイント 作成にローカル パスを使用できます。これは Artifacts としてアップロードされます。

```python
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## `WandbEvalCallback` を使用してモデル の 予測 を 可視化 する

{{< cta-button colabLink="https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}

`WandbEvalCallback` は、主にモデル の 予測 用に、次いで データセット の 可視化 用に Keras コールバック を構築するための抽象ベース クラス です。

この抽象コールバックは、 データセット とタスクに関して不可知論的です。これを使用するには、このベース `WandbEvalCallback` コールバック クラス から継承し、`add_ground_truth` および `add_model_prediction` メソッド を実装します。

`WandbEvalCallback` は、次の メソッド を提供するユーティリティ クラス です。

* データ と 予測 の `wandb.Table` インスタンス を作成します。
* データ と 予測 の Tables を `wandb.Artifact` として記録します。
* トレーニング開始時にデータ テーブル を記録します `on_train_begin`。
* エポック 終了時に 予測 テーブル を記録します `on_epoch_end`。

次の例では、画像分類タスクに `WandbClfEvalCallback` を使用します。この例のコールバックは、検証データ (`data_table`) を W&B に記録し、推論を実行し、すべての エポック の最後に 予測 (`pred_table`) を W&B に記録します。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback


# モデル の 予測 の 可視化 コールバック を実装します
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

# 新しい W&B の run を初期化します
wandb.init(config={"hyper": "parameter"})

# コールバック を Model.fit に追加します
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
W&B [Artifact ページ]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ja" >}}) には、デフォルトで **Workspace** ページ ではなく Table ログ が含まれています。
{{% /alert %}}

### `WandbEvalCallback` リファレンス

| パラメータ | 説明 |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (リスト) `data_table` の列名のリスト |
| `pred_table_columns` | (リスト) `pred_table` の列名のリスト |

### メモリー フットプリント の詳細

`on_train_begin` メソッド が呼び出されると、`data_table` を W&B に記録します。W&B Artifact としてアップロードされると、`data_table_ref` クラス 変数 を使用してアクセスできるこのテーブルへの参照を取得します。`data_table_ref` は、`self.data_table_ref[idx][n]` のようにインデックス を付けることができる 2D リスト です。ここで、`idx` は行番号、`n` は列番号です。以下の例で使用方法を見てみましょう。

### コールバック をカスタマイズする

`on_train_begin` または `on_epoch_end` メソッド をオーバーライドして、よりきめ細かい制御を行うことができます。`N` バッチ 後に サンプル を記録する場合は、`on_train_batch_end` メソッド を実装できます。

{{% alert %}}
💡 `WandbEvalCallback` を継承してモデル の 予測 の 可視化 用のコールバックを実装していて、明確にするか修正する必要がある場合は、[issue](https://github.com/wandb/wandb/issues) を開いてお知らせください。
{{% /alert %}}

## `WandbCallback` [レガシー]

W&B ライブラリ [`WandbCallback`]({{< relref path="/ref/python/integrations/keras/wandbcallback" lang="ja" >}}) クラス を使用して、`model.fit` で追跡されるすべての メトリクス と損失値を自動的に保存します。

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Keras でモデル を設定する コード

# コールバック を model.fit に渡します
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

短いビデオ [1 分以内に Keras と Weights & Biases を使い始める](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M) をご覧ください。

詳細なビデオについては、[Weights & Biases を Keras と統合する](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab_channel=Weights%26Biases) をご覧ください。[Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras_pipeline_with_Weights_and_Biases.ipynb) を確認できます。

{{% alert %}}
スクリプト については、[example repo](https://github.com/wandb/examples) を参照してください。[Fashion MNIST example](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py) と、それが生成する [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) が含まれています。
{{% /alert %}}

`WandbCallback` クラス は、 モニター する メトリクス の指定、重みと 勾配 の追跡、training_data と validation_data の 予測 のログ記録など、さまざまなログ構成オプションをサポートしています。

詳細については、[`keras.WandbCallback` のリファレンス ドキュメント]({{< relref path="/ref/python/integrations/keras/wandbcallback.md" lang="ja" >}}) を確認してください。

`WandbCallback`

* Keras によって収集された メトリクス から履歴データ を自動的にログ に記録します: 損失 と `keras_model.compile()` に渡されたもの。
* `monitor` および `mode` 属性 で定義されているように、最適なトレーニング ステップ に関連付けられた run の概要 メトリクス を設定します。これはデフォルトで、最小 `val_loss` の エポック になります。`WandbCallback` はデフォルトで、最適な `epoch` に関連付けられたモデル を保存します。
* オプションで、 勾配 と パラメータ のヒストグラム を記録します。
* オプションで、wandb で 可視化 するためのトレーニングと検証のデータ を保存します。

### `WandbCallback` リファレンス

| 引数 | |
| -------------------------- | ------------------------------------------- |
| `monitor` | (str) モニター する メトリクス の名前。デフォルトは `val_loss`。|
| `mode` | (str) {`auto`、`min`、`max`} のいずれか。`min` - モニター が最小化されたときにモデル を保存する `max` - モニター が最大化されたときにモデル を保存する `auto` - モデル を保存するタイミングを推測しようとします (デフォルト)。|
| `save_model` | True - モニター が以前のすべての エポック よりも優れている場合にモデル を保存する False - モデル を保存しない |
| `save_graph` | (boolean) True の場合、モデル グラフ を wandb に保存します (デフォルトは True)。|
| `save_weights_only` | (boolean) True の場合、モデル の重みのみを保存します (`model.save_weights(filepath)`)。それ以外の場合は、完全なモデル を保存します)。 |
| `log_weights` | (boolean) True の場合、モデル のレイヤー の重みのヒストグラム を保存します。|
| `log_gradients` | (boolean) True の場合、トレーニング 勾配 のヒストグラム をログ に記録します |
| `training_data` | (tuple) `model.fit` に渡されるのと同じ形式 `(X,y)`。これは 勾配 を計算するために必要です - `log_gradients` が `True` の場合、これは必須です。|
| `validation_data` | (tuple) `model.fit` に渡されるのと同じ形式 `(X,y)`。wandb が 可視化 するための データ のセット。このフィールドを設定すると、すべての エポック で、wandb は少数の 予測 を行い、後で 可視化 するために 結果 を保存します。|
| `generator` | (generator) wandb が 可視化 するための検証データ を返す ジェネレーター。この ジェネレーター は タプル `(X,y)` を返す必要があります。wandb が特定の データ の例を 可視化 するには、`validate_data` または ジェネレーター のいずれかを設定する必要があります。|
| `validation_steps` | (int) `validation_data` が ジェネレーター の場合、完全な検証セット に対して ジェネレーター を実行するステップ 数。|
| `labels` | (リスト) wandb で データ を 可視化 している場合、この ラベル のリストは、複数の クラス を持つ 分類器 を構築している場合に、数値出力を理解可能な文字列に変換します。バイナリ 分類器 の場合は、2 つの ラベル のリスト \[`false の ラベル`、`true の ラベル`] を渡すことができます。`validate_data` と `generator` が両方とも false の場合、これは何も行いません。|
| `predictions` | (int) 各 エポック で 可視化 する 予測 の数。最大は 100 です。|
| `input_type` | (string) 可視化 を支援するためのモデル 入力 のタイプ。(「image」、「images」、「segmentation_mask」) のいずれかになります。|
| `output_type` | (string) 可視化 を支援するためのモデル 出力 のタイプ。(「image」、「images」、「segmentation_mask」) のいずれかになります。|
| `log_evaluation` | (boolean) True の場合、各 エポック で検証データ とモデル の 予測 を含む Table を保存します。詳細については、`validation_indexes`、`validation_row_processor`、および `output_row_processor` を参照してください。|
| `class_colors` | ([float, float, float]) 入力 または 出力 が セグメンテーション マスク の場合、各 クラス の rgb タプル (範囲 0 ～ 1) を含む 配列。|
| `log_batch_frequency` | (integer) None の場合、コールバック はすべての エポック をログ に記録します。整数に設定すると、コールバック は `log_batch_frequency` バッチ ごとにトレーニング メトリクス をログ に記録します。|
| `log_best_prefix` | (string) None の場合、追加の概要 メトリクス は保存されません。文字列に設定すると、 モニター された メトリクス と エポック に プレフィックス が付加され、 結果 が概要 メトリクス として保存されます。|
| `validation_indexes` | ([wandb.data_types._TableLinkMixin]) 各検証例に関連付ける インデックス キー の順序付きリスト。`log_evaluation` が True で、`validation_indexes` を指定した場合、検証データ の Table は作成されません。代わりに、各 予測 を `TableLinkMixin` で表される行に関連付けます。行キー のリストを取得するには、`Table.get_index()` を使用します。|
| `validation_row_processor` | (Callable) 検証データ に適用する関数。通常は データ を 可視化 するために使用されます。この関数は `ndx` (int) と `row` (dict) を受け取ります。モデル に 単一 の 入力 がある場合、`row["input"]` には行の 入力 データ が含まれます。それ以外の場合は、入力 スロット の名前が含まれます。適合関数が 単一 の ターゲット を取る場合、`row["target"]` には行の ターゲット データ が含まれます。それ以外の場合は、 出力 スロット の名前が含まれます。たとえば、 入力 データ が 単一 の 配列 の場合、 データ を 画像 として 可視化 するには、プロセッサー として `lambda ndx, row: {"img": wandb.Image(row["input"])}` を指定します。`log_evaluation` が False であるか、`validation_indexes` が存在する場合は無視されます。|
| `output_row_processor` | (Callable) `validation_row_processor` と同じですが、モデル の 出力 に適用されます。`row["output"]` にはモデル 出力 の 結果 が含まれます。|
| `infer_missing_processors` | (Boolean) 欠落している場合に `validation_row_processor` と `output_row_processor` を推論するかどうかを決定します。デフォルトは True です。`labels` を指定すると、W&B は必要に応じて 分類 タイプのプロセッサー を推論しようとします。|
| `log_evaluation_frequency` | (int) 評価 結果 をログ に記録する頻度を決定します。デフォルトは `0` で、トレーニング の最後にのみログ に記録します。すべての エポック でログ に記録する場合は 1 に、他のすべての エポック でログ に記録する場合は 2 に設定します。`log_evaluation` が False の場合は効果がありません。|

## よくある質問

### `wandb` で `Keras` マルチプロセッシング を使用するにはどうすればよいですか?

`use_multiprocessing=True` を設定すると、次のエラーが発生する場合があります。

```python
Error("You must call wandb.init() before wandb.config.batch_size")
```

回避策:

1. `Sequence` クラス のコンストラクションで、`wandb.init(group='...')` を追加します。
2. `main` で、`if __name__ == "__main__":` を使用していることを確認し、スクリプト ロジック の残りをその中に入れます。
```