---
title: Wandb コールバック
menu:
  reference:
    identifier: ja-ref-python-integrations-keras-wandbcallback
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/keras.py#L291-L1091 >}}

`WandbCallback` は、keras と wandb を自動的に統合します。

```python
WandbCallback(
    monitor="val_loss", verbose=0, mode="auto", save_weights_only=(False),
    log_weights=(False), log_gradients=(False), save_model=(True),
    training_data=None, validation_data=None, labels=None, predictions=36,
    generator=None, input_type=None, output_type=None, log_evaluation=(False),
    validation_steps=None, class_colors=None, log_batch_frequency=None,
    log_best_prefix="best_", save_graph=(True), validation_indexes=None,
    validation_row_processor=None, prediction_row_processor=None,
    infer_missing_processors=(True), log_evaluation_frequency=0,
    compute_flops=(False), **kwargs
)
```

#### 例:

```python
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[WandbCallback()],
)
```

`WandbCallback` は、keras によって収集されたメトリクスからの履歴データを自動的にログします: 損失および `keras_model.compile()` に渡されたもの。

`WandbCallback` は、「最良」のトレーニングステップに関連付けられた run のサマリーメトリクスを設定します。「最良」は `monitor` と `mode` 属性によって定義されます。デフォルトでは、最小の `val_loss` を持つエポックです。`WandbCallback` はデフォルトで最良の `epoch` に関連するモデルを保存します。

`WandbCallback` は、勾配およびパラメータのヒストグラムをオプションでログすることができます。

`WandbCallback` は、wandb によるトレーニングおよび検証データの可視化のためにデータを保存することができます。

| 引数 |  |
| :--- | :--- |
|  `monitor` |  (str) 監視するメトリックの名前。デフォルトは `val_loss` です。 |
|  `mode` |  (str) {`auto`, `min`, `max`} のいずれかです。 `min` - 監視が最小化されたときにモデルを保存する `max` - 監視が最大化されたときにモデルを保存する `auto` - モデルを保存するタイミングを推測しようとする（デフォルト）。 |
|  `save_model` |  True - 監視がすべての以前のエポックを上回ったときにモデルを保存する False - モデルを保存しない |
|  `save_graph` |  (ブール) True の場合、モデルのグラフを wandb に保存する（デフォルトは True）。 |
|  `save_weights_only` |  (ブール) True の場合、モデルの重みのみが保存されます (`model.save_weights(filepath)`)、そうでなければ、完全なモデルが保存されます (`model.save(filepath)`)。 |
|  `log_weights` |  (ブール) True の場合、モデルのレイヤの重みのヒストグラムを保存します。 |
|  `log_gradients` |  (ブール) True の場合、トレーニング勾配のヒストグラムをログします。 |
|  `training_data` |  (タプル) `model.fit` に渡される形式 `(X,y)` と同じ形式です。勾配を計算するために必要です - `log_gradients` が `True` の場合は必須です。 |
|  `validation_data` |  (タプル) `model.fit` に渡される形式 `(X,y)` と同じ形式です。wandb が可視化するデータセットです。設定されている場合、各エポックで wandb は少数の予測を行い、後で可視化するためにその結果を保存します。画像データを扱っている場合は、正しくログするために `input_type` と `output_type` を設定して下さい。 |
|  `generator` |  (ジェネレータ) wandb が可視化するための検証データを返すジェネレータ。このジェネレータは、タプル `(X,y)` を返す必要があります。wandb が特定のデータ例を可視化するには `validate_data` またはジェネレータが設定されている必要があります。画像データを扱っている場合は、正しくログするために `input_type` と `output_type` を設定してください。 |
|  `validation_steps` |  (int) `validation_data` がジェネレータの場合、完全な検証セットのためにジェネレータを実行するステップ数。 |
|  `labels` |  (リスト) あなたのデータを wandb で可視化する場合、このラベルのリストは、数値出力を理解可能な文字列に変換します。多クラス分類器を構築している場合に役立ちます。バイナリ分類器を作成している場合は、2つのラベル ["false のラベル", "true のラベル"] のリストを渡すことができます。 `validate_data` とジェネレータが両方 false の場合は何もしません。 |
|  `predictions` |  (int) 各エポックで可視化のために行う予測の数、最大は 100。 |
|  `input_type` |  (文字列) 可視化を支援するためのモデルの入力のタイプ。次のいずれかです: (`image`, `images`, `segmentation_mask`, `auto`)。 |
|  `output_type` |  (文字列) 可視化を支援するためのモデルの出力のタイプ。次のいずれかです: (`image`, `images`, `segmentation_mask`, `label`)。 |
|  `log_evaluation` |  (ブール) True の場合、各エポックで検証データとモデルの予測を含む Table を保存します。詳細は `validation_indexes`、`validation_row_processor`、および `output_row_processor` を参照してください。 |
|  `class_colors` |  ([float, float, float]) 入力または出力がセグメンテーションマスクの場合、各クラスの rgb タプル（範囲は 0-1）を含む配列。 |
|  `log_batch_frequency` |  (整数) None の場合、コールバックは毎エポックでログを記録します。整数に設定すると、コールバックは `log_batch_frequency` バッチごとにトレーニングメトリクスをログします。 |
|  `log_best_prefix` |  (文字列) None の場合、追加のサマリーメトリクスは保存されません。文字列に設定すると、監視されているメトリックとエポックがこの値で前置され、サマリーメトリクスとして保存されます。 |
|  `validation_indexes` |  ([wandb.data_types._TableLinkMixin]) 各検証例に関連付けるインデックスキーの順序付きリスト。 `log_evaluation` が True で `validation_indexes` が提供されている場合、検証データの Table は作成されず、各予測は `TableLinkMixin` によって表される行と関連付けられます。これらのキーを取得する最も一般的な方法は `Table.get_index()` を使用することで、行キーのリストが返されます。 |
|  `validation_row_processor` |  (Callable) 検証データに適用する関数で、一般的にはデータを可視化するために使用されます。この関数は `ndx` (int) と `row` (dict) を受け取ります。あなたのモデルが単一の入力を持っている場合、`row["input"]` は行の入力データです。それ以外の場合は、入力スロットの名前に基づいてキー化されます。あなたの fit 関数が単一のターゲットを取る場合、`row["target"]` は行のターゲットデータです。それ以外の場合は、出力スロットの名前に基づいてキー化されます。例えば、入力データが単一の ndarray であり、データを画像として可視化したい場合、`lambda ndx, row: {"img": wandb.Image(row["input"])}` をプロセッサとして提供できます。 `log_evaluation` が False の場合または `validation_indexes` が存在する場合は無視されます。 |
|  `output_row_processor` |  (Callable) `validation_row_processor` と同様ですが、モデルの出力に適用されます。`row["output"]` はモデル出力の結果を含みます。 |
|  `infer_missing_processors` |  (bool) `validation_row_processor` および `output_row_processor` を欠けている場合に推測するかどうかを決定します。デフォルトは True です。`labels` が提供されている場合、適切な場合に分類タイプのプロセッサを推測しようとします。 |
|  `log_evaluation_frequency` |  (int) 評価結果がログされる頻度を決定します。デフォルトは 0 で（トレーニングの最後のみ）、1 に設定すると毎エポック、2 に設定すると隔エポックでログします。 `log_evaluation` が False の場合、効果はありません。 |
|  `compute_flops` |  (bool) あなたの Keras Sequential または Functional モデルの FLOPs を GigaFLOPs 単位で計算します。 |

## メソッド

### `get_flops`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/keras.py#L1045-L1091)

```python
get_flops() -> float
```

tf.keras.Model または tf.keras.Sequential モデルの推論モードでの FLOPS [GFLOPs] を計算します。

内部では tf.compat.v1.profiler を使用しています。

### `set_model`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/keras.py#L567-L576)

```python
set_model(
    model
)
```

### `set_params`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/keras.py#L564-L565)

```python
set_params(
    params
)
```