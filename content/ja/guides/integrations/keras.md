---
title: Keras
menu:
  default:
    identifier: keras
    parent: integrations
weight: 160
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases_keras.ipynb" >}}

## Keras コールバック

W&B には、`wandb` v0.13.4 以降で利用可能な Keras 用の 3 つのコールバックがあります。従来の `WandbCallback` については下までスクロールしてください。

- **`WandbMetricsLogger`** : [Experiment Tracking]({{< relref "/guides/models/track" >}}) のためのコールバックです。トレーニングや検証のメトリクス、システムメトリクスを W&B に自動でログします。

- **`WandbModelCheckpoint`** : モデルのチェックポイントを W&B [Artifacts]({{< relref "/guides/core/artifacts/" >}}) に記録したい場合に使用します。

- **`WandbEvalCallback`**: この基本コールバックは、モデルの予測を W&B [Tables]({{< relref "/guides/models/tables/" >}}) にログし、インタラクティブな可視化を可能にします。

これらの新しいコールバックは次の特徴があります：

* Keras の設計思想に準拠しています。
* すべての用途に 1 つのコールバック（`WandbCallback`）を使う場合の認知負荷を減らします。
* ユーザーがニッチなユースケースに対応したコールバックをサブクラス化しやすくなっています。

## `WandbMetricsLogger` で実験管理をトラッキング

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}

`WandbMetricsLogger` は、`on_epoch_end` や `on_batch_end` などのコールバックメソッドで引数として渡される Keras の `logs` 辞書を自動でログします。

次の情報をトラッキングします：

* `model.compile` で定義したトレーニング／検証メトリクス
* システム（CPU/GPU/TPU）メトリクス
* 学習率（固定値またはスケジューラいずれも対応）

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

| パラメータ | 説明 | 
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | (`epoch`, `batch`, または `int`): `epoch` なら各エポック終了ごと、`batch` なら各バッチ終了ごと、整数の場合にはそのバッチ数ごとにログします。デフォルトは `epoch`。                                 |
| `initial_global_step` | (int): トレーニングを `initial_epoch` から再開し、かつ学習率スケジューラを使う場合に、学習率を正しくログするための引数です。step_size * initial_step で計算できます。デフォルトは 0。 |

## `WandbModelCheckpoint` でモデルのチェックポイントを記録

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}

`WandbModelCheckpoint` コールバックを使うことで、Keras のモデル（`SavedModel` フォーマット）やモデルの重みを一定のタイミングで保存し、W&B の `wandb.Artifact` としてアップロードできます（モデルのバージョン管理に最適）。

このコールバックは [`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) を継承しているため、チェックポイント処理のロジックは親コールバックが担っています。

このコールバックで保存できるもの：

* 指定した monitor 基準で最も良いパフォーマンスを記録したモデル
* パフォーマンスに関係なく、各エポック終了時のモデル
* 指定したエポックやバッチ単位でのモデル
* モデルの重みだけ、または全モデル
* `SavedModel` フォーマットまたは `.h5` フォーマットのいずれか

`WandbMetricsLogger` と組み合わせて使うことを推奨します。

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

| パラメータ | 説明 |
| ------------------------- |  ---- | 
| `filepath`   | (str): モデルファイルを保存するパス |  
| `monitor`                 | (str): 監視するメトリック名         |
| `verbose`                 | (int): 冗長度モード。0 で非表示、1 でコールバックが動作する際にメッセージを表示   |
| `save_best_only`          | (Boolean): `save_best_only=True` の場合、定義した `monitor`・`mode` 基準で最新または最良のモデルのみ保存   |
| `save_weights_only`       | (Boolean): True なら重みだけ保存                                            |
| `mode`                    | (`auto`, `min`, `max`): 例：`val_acc` なら `max`, `val_loss` なら `min` など  |                     
| `save_freq`               | ("epoch" または int): `'epoch'` の場合は各エポック終了時、整数の場合は指定したバッチ数ごとに保存。`val_acc` や `val_loss` など検証メトリクスを監視する場合は "epoch" を推奨 |
| `options`                 | (str): `save_weights_only` が true ならオプションの `tf.train.CheckpointOptions` オブジェクト、それ以外は `tf.saved_model.SaveOptions` |
| `initial_value_threshold` | (float): 監視するメトリックの初期「最良」値       |

### N エポックごとにチェックポイントをアップロードする

デフォルト（`save_freq="epoch"`）では、各エポックの終了時にチェックポイントを作成し、アーティファクトとしてアップロードします。特定バッチ数ごとにチェックポイントを行いたい場合は `save_freq` に整数を指定してください。N エポックごとの場合、`train` データローダーのカーディナリティを計算して `save_freq` に渡します：

```python
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### TPU アーキテクチャーでの効率的なチェックポイント記録

TPU 上でチェックポイント処理時に `UnimplementedError: File system scheme '[local]' not implemented` というエラーが出ることがあります。これは、モデルディレクトリー（`filepath`）がクラウドストレージ（`gs://bucket-name/...`）を使う必要があるためです（このバケットに TPU サーバーからアクセスできることが前提）。ただし、一度ローカル保存したものを Artifacts としてアップロードする形で回避できます。

```python
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## `WandbEvalCallback` でモデル予測を可視化

{{< cta-button colabLink="https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}

`WandbEvalCallback` は、主にモデル予測、次いでデータセット可視化のために使う抽象基底クラスです。

データセットやタスクを問わず agnostic（非依存）な作りになっているため、このクラスを継承し、`add_ground_truth` と `add_model_prediction` メソッドを実装して利用します。

`WandbEvalCallback` は次のメソッドを提供します：

* データと予測の `wandb.Table` インスタンスを作成
* データと予測の Tables を `wandb.Artifact` としてログ
* `on_train_begin` タイミングでデータテーブルをログ
* `on_epoch_end` で予測テーブルをログ

以下の例では、画像分類用に `WandbClfEvalCallback` を作成しています。このコールバックは検証データ（`data_table`）を W&B にログし、推論結果（`pred_table`）も W&B に毎エポック記録します。

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback

# モデル予測可視化コールバックの実装例
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

# Model.fit にコールバックを追加
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
W&B の [Artifact ページ]({{< relref "/guides/core/artifacts/explore-and-traverse-an-artifact-graph" >}}) にはデフォルトで Table ログが含まれています（**Workspace** ページではなく）。
{{% /alert %}}

### `WandbEvalCallback` リファレンス

| パラメータ            | 説明                                      |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (list) `data_table` のカラム名リスト |
| `pred_table_columns` | (list) `pred_table` のカラム名リスト |

### メモリ使用量の詳細

`on_train_begin` メソッドが呼ばれるタイミングで、`data_table` を W&B にログします。アップロード後、そのテーブルへの参照（`data_table_ref`）が取得でき、これは `self.data_table_ref[idx][n]` の形で 2 次元リストのようにアクセスできます（`idx` が行番号、`n` が列番号となります）。下の例で使い方を確認できます。

### コールバックのカスタマイズ

より細かく制御したい場合は `on_train_begin` や `on_epoch_end` をオーバーライド可能です。`N` バッチごとにサンプルをログしたい場合は `on_train_batch_end` メソッドを実装してください。

{{% alert %}}
`WandbEvalCallback` を継承してモデル予測可視化用のコールバックを実装する際、何か不明点や修正希望がある場合は [issue](https://github.com/wandb/wandb/issues) をご連絡ください。
{{% /alert %}}

## `WandbCallback` [従来版]

W&B ライブラリの `WandbCallback` クラスを使うと、`model.fit` でトラッキングされたすべてのメトリクスや損失値を自動で保存できます。

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # モデルのセットアップコード

# コールバックを model.fit に渡す
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

[Get Started with Keras and W&B in Less Than a Minute](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M) というショート動画をご覧いただけます。

より詳細な動画はこちら：[Integrate W&B with Keras](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab_channel=Weights%26Biases)。[Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras_pipeline_with_Weights_and_Biases.ipynb) も参考にどうぞ。

{{% alert %}}
[example repo](https://github.com/wandb/examples) にはスクリプト例として [Fashion MNIST example](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py) や、生成される [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) もあります。
{{% /alert %}}

`WandbCallback` クラスは多彩なログ設定オプションをサポートしています：監視メトリクスの指定、重みや勾配のトラッキング、トレーニングや検証データの予測ログなどなど。

`keras.WandbCallback` のリファレンスドキュメントもぜひご覧ください。

`WandbCallback` の特徴

* Keras が収集した任意のメトリクス（損失や `keras_model.compile()` で渡したもの）を自動で履歴データとしてログ
* `monitor` と `mode` で定義された「最良」のトレーニングステップのサマリーメトリクスを Run にセット。デフォルトは最小 `val_loss` のエポック。最良 `epoch` のモデルを自動保存。
* オプションで勾配やパラメータのヒストグラムも可視化
* オプションでトレーニング・検証データも可視化用に保存

### `WandbCallback` リファレンス

| 引数                  |                                    |
| -------------------------- | ------------------------------------------- |
| `monitor`                  | (str) 監視するメトリクス名。デフォルトは `val_loss`                                                                   |
| `mode`                     | (str) {`auto`, `min`, `max`} のいずれか。`min` は監視が最小化した時、`max` は最大化した時、`auto` は自動推定（デフォルト）                                                                                                                                                |
| `save_model`               | True なら監視基準を上回るモデルを保存／False なら保存しない                                       |
| `save_graph`               | (boolean) True でモデルグラフも wandb に保存（デフォルト True）                                                           |
| `save_weights_only`        | (boolean) True なら重みのみ（`model.save_weights(filepath)`）保存。それ以外はフルモデル                                       |
| `log_weights`              | (boolean) True で各レイヤーの重みのヒストグラムも保存                                                |
| `log_gradients`            | (boolean) True でトレーニング勾配をヒストグラムで保存                                                       |
| `training_data`            | (tuple) `model.fit` に渡す `(X,y)` 形式のデータ。勾配計算用に `log_gradients` True 時は必須       |
| `validation_data`          | (tuple) `model.fit` に渡す `(X,y)` 形式のデータ。設定すると各エポック、wandb が少数の予測を行い、可視化用に保存          |
| `generator`                | (generator) wandb 用可視化データ用ジェネレータ。`(X,y)` タプルを返す。`validate_data` または generator のいずれか指定     |
| `validation_steps`         | (int) `validation_data` が generator の場合の検証セット合計ステップ数       |
| `labels`                   | (list) 可視化用ラベルリスト。数値出力を分類器に分かりやすくラベルへ。2クラスの場合は `[false 用ラベル, true 用ラベル]` で。validate_data と generator が両方 false の場合は何もしません。    |
| `predictions`              | (int) 各エポックの可視化用予測数（最大 100）    |
| `input_type`               | (string) 入力可視化用型: (`image`, `images`, `segmentation_mask`)  |
| `output_type`              | (string) 出力可視化用型: (`image`, `images`, `segmentation_mask`)    |
| `log_evaluation`           | (boolean) True なら各エポックごとに検証データ＆モデル予測を含む Table を保存。追加詳細は `validation_indexes`、`validation_row_processor`、`output_row_processor` 参照     |
| `class_colors`             | (\[float, float, float]) セグメンテーションマスク用。各クラスの RGB タプル (0–1 の範囲)                  |
| `log_batch_frequency`      | (integer) None ならエポックごとにログ、値を指定すればそのバッチ数ごとにトレーニングメトリクスをログ          |
| `log_best_prefix`          | (string) None なら追加サマリーメトリクス無し。文字列を指定すれば監視メトリクスとエポックにプレフィックス付加してサマリー保存   |
| `validation_indexes`       | (\[wandb.data_types._TableLinkMixin]) 各検証サンプルに紐付けるキーの順序リスト。`log_evaluation` True かつ `validation_indexes` 指定時、検証データ Table は作成せず各予測を該当行と関連付けます。行キー取得は `Table.get_index()` を |
| `validation_row_processor` | (Callable) 検証データ可視化用関数（ndx:int, row:dict）。単一入力なら `row["input"]` にデータあり。例えば Image 可視化なら `lambda ndx, row: {"img": wandb.Image(row["input"])}` など。`log_evaluation` が False または `validation_indexes` 指定時は無視   |
| `output_row_processor`     | (Callable) 予測出力用。`row["output"]` にモデル出力値。          |
| `infer_missing_processors` | (Boolean) `validation_row_processor` や `output_row_processor` 無指定時、自動推論するかどうか。デフォルトは True。`labels` を指定した場合は分類用プロセッサも自動判定      |
| `log_evaluation_frequency` | (int) 評価結果ログの頻度を指定。デフォルト 0（学習終了時のみ）。1 で毎エポック、2 で隔エポック、など。`log_evaluation` False 時は影響なし    |

## よくある質問

### `Keras` の multiprocessing を `wandb` と組み合わせて使うには？

`use_multiprocessing=True` を設定すると次のようなエラーが発生することがあります：

```python
Error("You must call wandb.init() before wandb.config.batch_size")
```

この場合の回避方法：

1. `Sequence` クラスのコンストラクタで `wandb.init(group='...')` を追加
2. `main` 内では `if __name__ == "__main__":` を使い、他のスクリプトロジックをその中にまとめて記載してください
